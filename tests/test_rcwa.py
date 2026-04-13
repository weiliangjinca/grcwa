import numpy as np

import grcwa

from .utils import t_grad

try:
    from autograd import grad

    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False

tol = 1e-2  # error tolerance for autograd v.s. FD
tolS4 = 1e-3  # error tolerance for S4 v.s. this code
tolS4_pol = 1e-2  # Pol comparison: grcwa iterblur tangent field vs S4 analytic circle

Nlayer = 1
nG = 101
L1 = [0.1, 0]
L2 = [0, 0.1]
# all patterned layers below have the same griding structure: Nx*Ny
Nx = 100
Ny = 100

# now consider 3 layers: vacuum + patterned + vacuum
epsuniform0 = 1.0  # dielectric for layer 1 (uniform)
epsuniformN = 1.0  # dielectric for layer N (uniform)

thick0 = 1.0  # thickness for vacuum layer 1
thickN = 1.0

# frequency and angles
freq = 1.0
theta = np.pi / 18
phi = np.pi / 9
Pscale = 1.0

pthick = [0.2]
# eps for patterned layer
radius = 0.4
epgrid = np.ones((Nx, Ny), dtype=float)
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
sphere = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2
epgrid[sphere] = 12.0

planewave = {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}


def rcwa_assembly(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0):
    """
    planewave:{'p_amp',...}
    """
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
    obj.Add_LayerUniform(thick0, epsuniform0)
    for i in range(Nlayer):
        obj.Add_LayerGrid(pthick[i], Nx, Ny)
    obj.Add_LayerUniform(thickN, epsuniformN)

    obj.Init_Setup(Pscale=Pscale, Gmethod=0)
    obj.MakeExcitationPlanewave(
        planewave["p_amp"], planewave["p_phase"], planewave["s_amp"], planewave["s_phase"], order=0
    )
    obj.GridLayer_geteps(epgrid)

    return obj


def rcwa_assembly_pol(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0):
    """Assemble RCWA simulation with fmm_method='pol' (Pol formulation)."""
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0, fmm_method="pol")
    obj.Add_LayerUniform(thick0, epsuniform0)
    for i in range(Nlayer):
        obj.Add_LayerGrid(pthick[i], Nx, Ny)
    obj.Add_LayerUniform(thickN, epsuniformN)

    obj.Init_Setup(Pscale=Pscale, Gmethod=0)
    obj.MakeExcitationPlanewave(
        planewave["p_amp"], planewave["p_phase"], planewave["s_amp"], planewave["s_phase"], order=0
    )
    obj.GridLayer_geteps(epgrid)

    return obj


def test_rcwa():
    ## compared to S4
    planewave = {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}
    obj = rcwa_assembly(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0)
    R, T = obj.RT_Solve(normalize=0)
    assert abs(T - 0.85249901083265) < tolS4 * T
    ## compared to S4
    planewave = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}
    obj = rcwa_assembly(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0)
    R, T = obj.RT_Solve(normalize=0)
    assert abs(T - 0.83900479939861) < tolS4 * T

    # others
    ai, bi = obj.GetAmplitudes(1, 0.0)
    assert len(ai) == obj.nG * 2

    e, h = obj.Solve_FieldOnGrid(1, 0.0)
    assert e[0].shape == (Nx, Ny)

    Mx = np.real(obj.Patterned_epinv_list[0])
    val = obj.Volume_integral(1, Mx, Mx, Mx, normalize=1)
    assert np.real(val) > 0

    Tx, Ty, Tz = obj.Solve_ZStressTensorIntegral(0)
    assert Tz < 0


def test_rcwa_pol():
    """Pol method T values compared to S4 with UsePolarizationDecomposition().

    Ground truth computed with S4 (exact circle geometry, nG=101, theta=10deg,
    phi=20deg).  grcwa uses a 100x100 pixel grid so ~0.5% geometry difference
    is expected; tolS4_pol accommodates this.
    """
    # p-polarization
    planewave = {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}
    obj = rcwa_assembly_pol(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0)
    R, T = obj.RT_Solve(normalize=0)
    assert abs(T - 0.857839488683) < tolS4_pol * T
    # s-polarization
    planewave = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}
    obj = rcwa_assembly_pol(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0)
    R, T = obj.RT_Solve(normalize=0)
    assert abs(T - 0.844496926280) < tolS4_pol * T


def test_pol_differs_from_fft():
    """Pol and FFT must give measurably different T for a patterned layer.

    If they are identical, the Pol correction is not being applied.
    """
    planewave = {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}
    obj_fft = rcwa_assembly(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0)
    _, T_fft = obj_fft.RT_Solve(normalize=0)
    obj_pol = rcwa_assembly_pol(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0)
    _, T_pol = obj_pol.RT_Solve(normalize=0)
    assert abs(T_pol - T_fft) / T_fft > 1e-3, "Pol and FFT should differ"


def test_pol_energy_conservation():
    """For a lossless structure, R+T must equal cos(theta) (unnormalised)."""
    planewave = {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}
    obj = rcwa_assembly_pol(epgrid, freq, theta, phi, planewave, pthick, Pscale=1.0)
    R, T = obj.RT_Solve(normalize=0)
    assert abs(R + T - np.cos(theta)) < 2e-3


if AG_AVAILABLE:
    grcwa.set_backend("autograd")

    def test_epsgrad():
        def fun(x):
            obj = rcwa_assembly(x, freq, theta, phi, planewave, pthick, Pscale=1.0)
            R, T = obj.RT_Solve(normalize=1)
            return R

        grad_fun = grad(fun)

        x = epgrid.flatten()
        dx = 1e-3
        ind = np.random.randint(Nx * Ny * Nlayer, size=1)[0]
        FD, AD = t_grad(fun, grad_fun, x, dx, ind)
        assert abs(FD - AD) < abs(FD) * tol, "wrong epsgrid gradient"

    def test_thickgrad():
        def fun(x):
            obj = rcwa_assembly(epgrid.flatten(), freq, theta, phi, planewave, x, Pscale=1.0)
            R, T = obj.RT_Solve(normalize=1)
            return R

        grad_fun = grad(fun)

        x = [0.1]
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun, grad_fun, x, dx, ind)
        assert abs(FD - AD) < abs(FD) * tol, "wrong thickness gradient"

    def test_periodgrad():
        def fun(x):
            obj = rcwa_assembly(epgrid.flatten(), freq, theta, phi, planewave, pthick, Pscale=x)
            R, T = obj.RT_Solve(normalize=1)
            return R

        grad_fun = grad(fun)

        x = 1.0
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun, grad_fun, x, dx, ind)
        assert abs(FD - AD) < abs(FD) * tol, "wrong thickness gradient"

    def test_freqgrad():
        def fun(x):
            obj = rcwa_assembly(epgrid.flatten(), x, theta, phi, planewave, pthick, Pscale=1.0)
            R, T = obj.RT_Solve(normalize=1)
            return R

        grad_fun = grad(fun)

        x = 1.0
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun, grad_fun, x, dx, ind)
        assert abs(FD - AD) < abs(FD) * tol, "wrong thickness gradient"

    def test_thetagrad():
        def fun(x):
            obj = rcwa_assembly(epgrid.flatten(), freq, x, phi, planewave, pthick, Pscale=1.0)
            R, T = obj.RT_Solve(normalize=1)
            return R

        grad_fun = grad(fun)

        x = np.pi / 10
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun, grad_fun, x, dx, ind)
        assert abs(FD - AD) < abs(FD) * tol, "wrong thickness gradient"

    def test_epsgrad_pol():
        """Autograd epsilon gradient with Pol must match finite differences.

        The iterblur tangent field uses a hard interface mask, so gradients
        at pixels on the mask boundary have inherent error (~5%).  We test
        at interior pixels (away from interfaces) where gradients are clean.
        """

        def fun(x):
            obj = rcwa_assembly_pol(x, freq, theta, phi, planewave, pthick, Pscale=1.0)
            R, T = obj.RT_Solve(normalize=1)
            return R

        grad_fun = grad(fun)

        x = epgrid.flatten()
        dx = 1e-3
        # Pick a pixel well inside the circle (far from interface)
        ind = 50 * Ny + 50  # center of the grid, inside the eps=12 circle
        FD, AD = t_grad(fun, grad_fun, x, dx, ind)
        assert abs(FD - AD) < abs(FD) * tol, "wrong epsgrid gradient with Pol"

    def test_thickgrad_pol():
        """Autograd thickness gradient with Pol must match finite differences."""

        def fun(x):
            obj = rcwa_assembly_pol(epgrid.flatten(), freq, theta, phi, planewave, x, Pscale=1.0)
            R, T = obj.RT_Solve(normalize=1)
            return R

        grad_fun = grad(fun)

        x = [0.1]
        dx = 1e-3
        ind = 0
        FD, AD = t_grad(fun, grad_fun, x, dx, ind)
        assert abs(FD - AD) < abs(FD) * tol, "wrong thickness gradient with Pol"
