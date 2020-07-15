=====
Usage
=====

To use grcwa in a project::

    import grcwa

To enable autograd::

  grcwa.set_backend('autograd')

To initialize the RCWA::

  obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=0) # verbose=1 for output the actual nG

To add layers, the order of adding will determine the layer order (1st added layer is 0-th layer, 2nd to be 1st layer, and so forth)::
  
  obj.Add_LayerUniform(thick0,ep0) # uniform slab
  obj.Add_LayerGrid(thickp,Nx,Ny) # patterned layer

  # after add all layers:
  obj.Init_Setup()

To feed the epsilon profile for patterned layer::

  # x is a 1D array: np.concatenate((epgrid1.flatten(),epgrid2.flatten(),...))
  obj.GridLayer_geteps(x)

To scale the periodicity in the both lateral directions simultaneously (as an autogradable parameter)::

  obj.Init_Setup(Pscale=scale) # period will be scale*Lx and scale*Ly

Fourier space truncation options ::

  obj.Init_Setup(Gmethod=0) # 0 for circular, 1 for rectangular

To define planewave excitation::

  obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)

To define incidence light other than planewave::

  obj.a0 = ... # forward
  obj.bN = ... # backward, each have a length 2*obj.nG, for the 2 lateral directions
  
To normalize output when the 0-th media is not vacuum, or for oblique incidence::
  
  R, T = obj.RT_Solve(normalize = 1)

To get Poynting flux by order::
  
  Ri, Ti = obj.RT_Solve(byorder=1) # Ri(Ti) has length obj.nG, too see which order, check obj.G; too see which kx,ky, check obj.kx obj.ky

To get amplitude of eigenvectors at some layer at some zoffset ::

  ai,bi = obj.GetAmplitudes(which_layer,z_offset)


To get real-space epsilon profile reconstructured from the truncated Fourier orders ::
  
  ep = obj.Return_eps(which_layer,Nx,Ny,component='xx') # For patterned layer component = 'xx','xy','yx','yy','zz'; For uniform layer, currently it's assumed to be isotropic        
        
To get Fourier amplitude of fields at some layer at some zoffset ::

  E,H = obj.Solve_FieldFourier(which_layer,z_offset) #E = [Ex,Ey,Ez], H = [Hx,Hy,Hz]

To get fields in real space on grid points ::
  
  E,H = obj.Solve_FieldOnGrid(which_layer,z_offset) # #E = [Ex,Ey,Ez], H = [Hx,Hy,Hz]
  
To get volume integration with respect to some convolution matrix *M* defined for 3 directions, respectively::
  
  val = obj.Volume_integral(which_layer,Mx,My,Mz,normalize=1)

To compute Maxwell stress tensor, integrated over the *z*-plane::

  Tx,Ty,Tz = obj.Solve_ZStressTensorIntegral(which_layer)
