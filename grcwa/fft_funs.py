import numpy as np

from . import backend as bd

# Threshold for detecting material interfaces in the Pol tangent field.
# Gradients below this are treated as uniform (no interface).
POL_GRAD_TOL = np.finfo(np.float32).eps  # ~1.19e-7


def Epsilon_fft(dN, eps_grid, G):
    """dN = 1/Nx/Ny
    For now, assume epsilon is diagonal; if epsilon has xz,yz component, just simply add them to off-diagonal eps2

    eps_grid is  (1) for isotropic, a numpy 2d array in the format of (Nx,Ny),
                 (2) for anisotropic, a list of numpy 2d array [(Nx,Ny),(Nx,Ny),(Nx,Ny)]
    """

    if len(eps_grid) == 3 and eps_grid[0].ndim == 2:
        epsx_fft = get_conv(dN, eps_grid[0], G)
        epsy_fft = get_conv(dN, eps_grid[1], G)
        epsz_fft = get_conv(dN, eps_grid[2], G)
        epsinv = bd.inv(epsz_fft)

        tmp1 = bd.vstack((epsx_fft, bd.zeros_like(epsx_fft)))
        tmp2 = bd.vstack((bd.zeros_like(epsx_fft), epsy_fft))
        eps2 = bd.hstack((tmp1, tmp2))

    elif eps_grid[0].ndim == 1:
        eps_fft = get_conv(dN, eps_grid, G)
        epsinv = bd.inv(eps_fft)

        tmp1 = bd.vstack((eps_fft, bd.zeros_like(eps_fft)))
        tmp2 = bd.vstack((bd.zeros_like(eps_fft), eps_fft))
        eps2 = bd.hstack((tmp1, tmp2))
    else:
        raise ValueError("Wrong eps_grid type")

    return epsinv, eps2


def Epsilon_fft_pol(dN, eps_grid, G, pol_sigma=3.0, pol_niter=20):
    """Fourier-space epsilon matrices using the Pol method (S4 paper Eq. 51).

    Implements the PolBasisVL formulation from S4 (fmm_PolBasisVL.cpp), which
    applies proper Fourier factorization rules (Li 1996/1997) using a tangent
    vector field at material interfaces.  This converges significantly faster
    than the naive FFT/Laurent formulation (Epsilon_fft) for structures with
    discontinuous permittivity.

    The two returned matrices feed into the RCWA eigensolve exactly like
    Epsilon_fft.  The method is fully compatible with HIPS/autograd for
    gradient-based topology optimization.

    Parameters
    ----------
    dN : float
        1/(Nx*Ny), normalisation constant for the Fourier transform.
    eps_grid : 2d array (Nx, Ny)
        Isotropic permittivity on the real-space grid.  May be an autograd
        ArrayBox for automatic differentiation.
    G : array (nG, 2)
        Integer G-vector indices (Lk1, Lk2 components).
    pol_sigma : float
        Gaussian blur sigma (in pixels) for smoothing the tangent field.
    pol_niter : int
        Number of blur+reset iterations.  0 means a single blur (no reset).

    Returns
    -------
    epsinv : (nG, nG) complex array
        Inverse-rule Toeplitz matrix of 1/eps (used for the kp matrix).
    eps2 : (2*nG, 2*nG) complex array
        Pol-corrected in-plane epsilon matrix (block 2x2: xx, xy, yx, yy).

    References
    ----------
    V. Liu & S. Fan, Comp. Phys. Comm. 183, 2233 (2012), Eq. 51.
    """

    # --- epsinv: inverse rule for Ez (always normal to layers) ---
    inveps_grid = 1.0 / eps_grid
    eta_hat = get_conv(dN, inveps_grid, G)  # Toeplitz(1/eps)
    epsinv = eta_hat

    # --- eps2: Pol-corrected in-plane epsilon ---
    eps_hat = get_conv(dN, eps_grid, G)  # Toeplitz(eps)

    # Tangent-field projection operators P_ij (autograd-compatible)
    P_xx, P_xy, P_yx, P_yy = _compute_tangent_field_pol(
        eps_grid, pol_sigma=pol_sigma, pol_niter=pol_niter
    )

    # Fourier-transform the projection operators into (nG, nG) matrices
    P_xx_hat = get_conv(dN, P_xx, G)
    P_xy_hat = get_conv(dN, P_xy, G)
    P_yx_hat = get_conv(dN, P_yx, G)
    P_yy_hat = get_conv(dN, P_yy, G)

    # S4 sign convention (fmm_PolBasisVL.cpp lines 257-268):
    #   mDelta = inv(Eta) - Epsilon       (negated Delta in Eq. 51)
    #   eps2[block] += mDelta @ P[block]
    # Block order: (0,0)=xx, (0,1)=xy, (1,0)=yx, (1,1)=yy
    mDelta = bd.inv(eta_hat) - eps_hat

    E_xx = eps_hat + bd.dot(mDelta, P_xx_hat)
    E_xy = bd.dot(mDelta, P_xy_hat)
    E_yx = bd.dot(mDelta, P_yx_hat)
    E_yy = eps_hat + bd.dot(mDelta, P_yy_hat)

    top = bd.hstack((E_xx, E_xy))
    bot = bd.hstack((E_yx, E_yy))
    eps2 = bd.vstack((top, bot))

    return epsinv, eps2


def _compute_tangent_field_pol(eps_grid, pol_sigma=3.0, pol_niter=20):
    """Tangent vector field and projection operators for the Pol method.

    Computes a smooth tangent vector field at material interfaces and returns
    the outer-product projection matrices P_ij = t_i * t_j used by
    Epsilon_fft_pol.  The implementation is fully autograd-compatible so that
    gradients propagate correctly through the Pol correction during
    topology optimization.

    Algorithm:
      1. Detect interfaces via forward finite differences of eps (periodic BC).
      2. Rotate the gradient 90 deg to obtain the raw tangent field.
      3. Iterative blur+reset: blur the tangent field, then reset interface
         pixels to their exact gradient values.  Repeated pol_niter times,
         this approximates a Laplace solve (smooth harmonic extension from
         interface pixels).  If pol_niter=0, a single blur is applied with
         no reset (original behaviour).
      4. Normalize so max|t| = 1 (Pol scaling, as in S4).
      5. Form P_ij = t_i * t_j (no division by |t|^2).

    All operations use the ``bd`` backend so autograd can differentiate
    through them.  The only raw-numpy operations are the Gaussian kernel
    and interface mask (fixed constants) and the early-exit checks.

    Parameters
    ----------
    eps_grid : 2d array (Nx, Ny)
        Isotropic permittivity grid.  May be an autograd ArrayBox.
    pol_sigma : float
        Gaussian blur sigma in pixels for smoothing the tangent field.
    pol_niter : int
        Number of blur+reset iterations.  0 means a single blur (no reset).

    Returns
    -------
    P_xx, P_xy, P_yx, P_yy : 2d arrays (Nx, Ny)
        Projection-operator components.  P_yx == P_xy (symmetric).
        Autograd-tracked when eps_grid is tracked.
    """
    Nx = eps_grid.shape[0]
    Ny = eps_grid.shape[1]

    eps_re = bd.real(eps_grid)
    eps_im = bd.imag(eps_grid)

    # --- 1. Interface detection: periodic forward differences ---
    # Check both real and imaginary parts; an interface exists if either differs.
    grad_x_re = bd.concatenate([eps_re[1:, :], eps_re[:1, :]], axis=0) - eps_re
    grad_y_re = bd.concatenate([eps_re[:, 1:], eps_re[:, :1]], axis=1) - eps_re
    grad_x_im = bd.concatenate([eps_im[1:, :], eps_im[:1, :]], axis=0) - eps_im
    grad_y_im = bd.concatenate([eps_im[:, 1:], eps_im[:, :1]], axis=1) - eps_im

    # Early exit for uniform permittivity (detached check on both parts)
    def _detach(arr):
        return np.real(np.asarray(arr._value if hasattr(arr, "_value") else arr))

    _max_grad = np.sqrt(
        np.max(
            _detach(grad_x_re) ** 2
            + _detach(grad_y_re) ** 2
            + _detach(grad_x_im) ** 2
            + _detach(grad_y_im) ** 2
        )
    )
    if _max_grad < POL_GRAD_TOL:
        z = bd.zeros_like(eps_re)
        return z, z, z, z

    # Combined gradient: sum of real and imaginary contributions
    grad_x = grad_x_re + grad_x_im
    grad_y = grad_y_re + grad_y_im

    # --- 2. Tangent = 90-deg rotated gradient: t = (-grad_y, grad_x) ---
    tx_raw = -grad_y
    ty_raw = grad_x

    # --- 3. Iterative blur+reset to extend the tangent field smoothly ---
    # The blur kernel and interface mask are constant numpy arrays (not
    # differentiated).  The iteration approximates ∇²t = 0 with Dirichlet
    # BC at interface pixels, producing a smooth harmonic extension.
    kx_freq = np.fft.fftfreq(Nx)
    ky_freq = np.fft.fftfreq(Ny)
    KX, KY = np.meshgrid(kx_freq, ky_freq, indexing="ij")
    blur_kernel = np.exp(-2 * np.pi**2 * pol_sigma**2 * (KX**2 + KY**2))

    if pol_niter <= 0:
        # Single blur, no reset (original behaviour).
        tx = bd.real(bd.ifft2(bd.fft2(tx_raw) * blur_kernel))
        ty = bd.real(bd.ifft2(bd.fft2(ty_raw) * blur_kernel))
    else:
        # Interface mask: constant (detached) binary array.
        _grad_mag = np.sqrt(
            _detach(grad_x_re) ** 2
            + _detach(grad_y_re) ** 2
            + _detach(grad_x_im) ** 2
            + _detach(grad_y_im) ** 2
        )
        mask = (_grad_mag > POL_GRAD_TOL).astype(float)
        mask_inv = 1.0 - mask

        tx = tx_raw
        ty = ty_raw
        for _ in range(pol_niter):
            tx = mask * tx_raw + mask_inv * bd.real(bd.ifft2(bd.fft2(tx) * blur_kernel))
            ty = mask * ty_raw + mask_inv * bd.real(bd.ifft2(bd.fft2(ty) * blur_kernel))

    # --- 4. Pol scaling: max|t| = 1 ---
    t_mag_sq = tx * tx + ty * ty
    # Detached check only -- the normalisation itself is differentiable.
    _raw_mag = np.real(np.asarray(t_mag_sq._value if hasattr(t_mag_sq, "_value") else t_mag_sq))
    if np.max(_raw_mag) < POL_GRAD_TOL**2:
        z = bd.zeros_like(eps_re)
        return z, z, z, z
    max_field = bd.sqrt(bd.max(t_mag_sq))
    tx = tx / max_field
    ty = ty / max_field

    # --- 5. Projection operators ---
    P_xx = tx * tx
    P_xy = tx * ty
    P_yy = ty * ty

    return P_xx, P_xy, P_xy, P_yy


def get_conv(dN, s_in, G):
    """Attain convolution matrix
    dN = 1/Nx/Ny
    s_in: np.array of length Nx*Ny
    G: shape (nG,2), 2 for Lk1,Lk2
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)

    Auto-upsamples via nearest-neighbor when the grid is too coarse
    for the requested G-vectors (prevents FFT index wrap-around).
    """
    nG, _ = G.shape
    Nx = s_in.shape[0]
    Ny = s_in.shape[1]

    # Check if grid is large enough for the G-vector differences
    max_gdiff_x = int(np.max(G[:, 0])) - int(np.min(G[:, 0]))
    max_gdiff_y = int(np.max(G[:, 1])) - int(np.min(G[:, 1]))

    if max_gdiff_x >= Nx or max_gdiff_y >= Ny:
        # Upsample via nearest-neighbor (autograd-safe integer-division indexing)
        scale_x = max(int(np.ceil((max_gdiff_x + 1) / Nx)), 1)
        scale_y = max(int(np.ceil((max_gdiff_y + 1) / Ny)), 1)
        idx_x = np.arange(Nx * scale_x) // scale_x
        idx_y = np.arange(Ny * scale_y) // scale_y
        s_work = s_in[idx_x][:, idx_y]
        dN_work = 1.0 / (Nx * scale_x) / (Ny * scale_y)
    else:
        s_work = s_in
        dN_work = dN

    sfft = bd.fft2(s_work) * dN_work

    ix = range(nG)
    ii, jj = bd.meshgrid(ix, ix, indexing="ij")
    s_out = sfft[G[ii, 0] - G[jj, 0], G[ii, 1] - G[jj, 1]]
    return s_out


def get_fft(dN, s_in, G):
    """
    FFT to get Fourier components

    s_in: np.2d array of size (Nx,Ny)
    G: shape (nG,2), 2 for Gx,Gy
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    """

    sfft = bd.fft2(s_in) * dN
    return sfft[G[:, 0], G[:, 1]]


def get_ifft(Nx, Ny, s_in, G):
    """
    Reconstruct real-space fields
    """
    dN = 1.0 / Nx / Ny
    nG, _ = G.shape

    s0 = bd.zeros((Nx, Ny), dtype=complex)
    for i in range(nG):
        x = G[i, 0]
        y = G[i, 1]

        stmp = bd.zeros((Nx, Ny), dtype=complex)
        stmp[x, y] = 1.0
        s0 = s0 + s_in[i] * stmp

    s_out = bd.ifft2(s0) / dN
    return s_out
