import numpy as np
import grcwa
from .utils import t_grad

try:
    import autograd.numpy as npa
    from autograd import grad
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False
    
L1 = [0.5,0]
L2 = [0,0.2]
nG = 100
method = 0
kx0 = 0.1
ky0 = 0.2

Lk1,Lk2 = grcwa.Lattice_Reciprocate(L1,L2)
G,nGout = grcwa.Lattice_getG(nG,Lk1,Lk2,method=method)
kx, ky = grcwa.Lattice_SetKs(G, kx0, ky0, Lk1, Lk2)
    
def test_bloch():
    assert nGout>0,'negative nG'
    assert nGout<=nG,'wrong nG'

if AG_AVAILABLE:
    grcwa.set_backend('autograd')
    Nx = 51
    Ny = 71
    dN = 1./Nx/Ny
    tol = 1e-2    
    
    def test_fft():
        def fun(ep):
            epout = npa.reshape(ep,(Nx,Ny))
            epsinv, eps2 = grcwa.Epsilon_fft(dN,epout,G)
            return npa.real(npa.sum(epsinv))

        grad_fun = grad(fun)

        x = 1.+10.*np.random.random(Nx*Ny)
        dx = 1e-3
        ind = np.random.randint(Nx*Ny,size=1)[0]        
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong fft gradient'

    def test_fft_aniso():
        def fun(ep):
            epout = [npa.reshape(ep[x*Nx*Ny:(x+1)*Nx*Ny],(Nx,Ny)) for x in range(3)]
            epsinv, eps2 = grcwa.Epsilon_fft(dN,epout,G)
            return npa.real(npa.sum(eps2))

        grad_fun = grad(fun)

        x = 1.+10.*np.random.random(3*Nx*Ny)
        dx = 1e-3
        ind = np.random.randint(Nx*Ny*2,size=1)[0]        
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong fft gradient'        

    def test_ifft():
        ix = np.random.randint(Nx,size=1)[0]
        iy = np.random.randint(Ny,size=1)[0]
        def fun(x):
            out = grcwa.get_ifft(Nx,Ny,x,G)
            return npa.real(out[ix,iy])

        grad_fun = grad(fun)

        x = 10.*np.random.random(nGout)
        dx = 1e-3
        ind = np.random.randint(nGout,size=1)[0]
        FD, AD = t_grad(fun,grad_fun,x,dx,ind)
        assert abs(FD-AD)<abs(FD)*tol,'wrong ifft gradient'        
