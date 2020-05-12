"""Transmission and reflection of two patterned layers"""

import grcwa
import numpy as np
		    
# lattice constants
L1 = [0.2,0]
L2 = [0,0.2]
# Truncation order (actual number might be smaller)
nG = 101
# frequency
freq = 1.
# angle
theta = np.pi/10
phi = 0.

# setup RCWA
obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=1)

Np = 2 # number of patterned layers
Nx = 100
Ny = 100
		    
thick0 = 0.1
pthick = [0.2,0.3]
thickN = 0.4

ep0 = 2.
epN = 3.

# input layer information
obj.Add_LayerUniform(thick0,ep0)
for i in range(Np):
    obj.Add_LayerGrid(pthick[i],Nx,Ny)
obj.Add_LayerUniform(thickN,epN)

obj.Init_Setup()

# patterned layers
radius = 0.5
a = 0.5

ep1 = 4.
ep2 = 6.
epbkg = 1.

# coordinate
x0 = np.linspace(0,1.,Nx)
y0 = np.linspace(0,1.,Ny)
x, y = np.meshgrid(x0,y0,indexing='ij')

# layer 1
epgrid1 = np.ones((Nx,Ny))*ep1
ind = (x-.5)**2+(y-.5)**2<radius**2
epgrid1[ind]=epbkg

# layer 2
epgrid2 = np.ones((Nx,Ny))*ep2
ind = np.logical_and(np.abs(x-.5)<a/2, np.abs(y-.5)<a/2)
epgrid2[ind]=epbkg		    
		    
# combine epsilon of all layers
epgrid = np.concatenate((epgrid1.flatten(),epgrid2.flatten()))
obj.GridLayer_geteps(epgrid)


planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

# solve for R and T
R,T= obj.RT_Solve(normalize=1)                    
print('R=',R,', T=',T,', R+T=',R+T)
