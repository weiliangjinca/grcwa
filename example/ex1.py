"""Transmission and reflection of a square lattice of a hole."""
import grcwa
import numpy as np

# Truncation order (actual number might be smaller)
nG = 101
# lattice constants
L1 = [0.2,0]
L2 = [0,0.2]
# frequency and angles
freq = 1.
theta = 0.
phi = 0.
# the patterned layer has a griding: Nx*Ny
Nx = 100
Ny = 100

# now consider 3 layers: vacuum + patterned + vacuum
ep0 = 1. # dielectric for layer 1 (uniform)
epp = 10. # dielectric for patterned layer
epbkg = 1. # dielectric for holes in the patterned layer 
epN = 1.  # dielectric for layer N (uniform)

thick0 = 1. # thickness for vacuum layer 1
thickp = 0.2 # thickness of patterned layer
thickN = 1.

# eps for patterned layer
radius = 0.4
epgrid = np.ones((Nx,Ny),dtype=float)*epp
x0 = np.linspace(0,1.,Nx)
y0 = np.linspace(0,1.,Ny)
x, y = np.meshgrid(x0,y0,indexing='ij')
sphere = (x-.5)**2+(y-.5)**2<radius**2
epgrid[sphere] = epbkg

######### setting up RCWA
obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=1)
# input layer information
obj.Add_LayerUniform(thick0,ep0)
obj.Add_LayerGrid(thickp,Nx,Ny)
obj.Add_LayerUniform(thickN,epN)
obj.Init_Setup()

# planewave excitation
planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
# eps in patterned layer
obj.GridLayer_geteps(epgrid.flatten())

# compute reflection and transmission
R,T= obj.RT_Solve(normalize=1)

print('R=',R,', T=',T,', R+T=',R+T)
