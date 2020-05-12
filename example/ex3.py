""" Topology optimization of reflection of a single patterned layers ."""
""" Nlopt is needed """

import grcwa
grcwa.set_backend('autograd')  # important!!

import numpy as npf
import autograd.numpy as np
from autograd import grad

try:
    import nlopt
    NL_AVAILABLE = True
except ImportError:
    NL_AVAILABLE = False

if NL_AVAILABLE == False:
    raise Exception('Please install NLOPT')

# Truncation order (actual number might be smaller)
nG = 101
# lattice constants
L1 = [0.5,0]
L2 = [0,0.5]
# frequency and angles
freq = 1.
theta = 0.
phi = 0.
# the patterned layer has a griding: Nx*Ny
Nx = 100
Ny = 100

# now consider 3 layers: vacuum + patterned + vacuum
ep0 = 1. # dielectric for layer 1 (uniform)
epp = 12. # dielectric for patterned layer
epbkg = 1. # dielectric for holes in the patterned layer 
epN = 1.  # dielectric for layer N (uniform)

thick0 = 1. # thickness for vacuum layer 1
thickp = 0.5 # thickness of patterned layer
thickN = 1.
# planewave excitation
planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}


# set up objective function, x is the dielctric constant on the 2D grids, of size Nx*Ny
# Qabs is a parameter for relatxation to better approach global optimal, at Qabs = inf, it will describe the real physics.
# It also be used to resolve the singular matrix issues by setting a large but finite Qabs, e.g. Qabs = 1e5

def fun_reflection(x,Qabs):
    freqcmp = freq*(1+1j/2/Qabs)
    ######### setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    # input layer information
    obj.Add_LayerUniform(thick0,ep0)
    obj.Add_LayerGrid(thickp,Nx,Ny)
    obj.Add_LayerUniform(thickN,epN)
    obj.Init_Setup()

    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)    
    obj.GridLayer_geteps(x)
    R,T= obj.RT_Solve(normalize=1)
    return R


# nlopt function
ctrl = 0
Qabs = np.inf
fun = lambda x: fun_reflection(x,Qabs)
grad_fun = grad(fun)
def fun_nlopt(x,gradn):
    global ctrl
    gradn[:] = grad_fun(x)
    y = fun(x)
    
    print('Step = ',ctrl,', R = ',y)
    ctrl += 1
    return fun(x)

# set up NLOPT
ndof = Nx*Ny
init = epbkg+ (epp-epbkg)*np.random.random(ndof)
lb=np.ones(ndof,dtype=float)*epbkg
ub=np.ones(ndof,dtype=float)*epp

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(100)

opt.set_max_objective(fun_nlopt)
x = opt.optimize(init)
