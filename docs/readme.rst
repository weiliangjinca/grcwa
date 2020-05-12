========
Tutorial
========

* Installation:

  .. code-block:: console
		  
		  $ pip install grcwa

  Or,

  .. code-block:: console

		  $ git clone git://github.com/weiliangjinca/grcwa
		  $ pip install .

* Example 1: transmission and reflection of a square lattice of a hole: see *ex1.py* in the example folder.

* Example 2: Transmission and reflection of two patterned layers: (see *ex2.py* in the example folder), as illustrated in the figure below (only a **unit cell** is plotted)

  .. image:: ex.png
	     
  * *Periodicity* in the lateral direction is  *L*\ :sub:`x` = *L*\ :sub:`y` = 0.2, and *frequency* is 1.0.

  * The incident light has an angel *pi*/10.

    .. code-block:: python
		  
		    import grcwa
		    import numpy as np
		    grcwa.set_backend('autograd') # if autograd needed
		    
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

  * Geometry: the thicknesses of the four layers are 0.1,0.2,0.3, and 0.4. For patterned layers, we consider total grid points *N*\ :sub:`x` \* *N*\ :sub:`y` = 100\*100 within the unit cell.
    
  * Dielectric constant: 2.0 for the 0-th layer; 4.0 (1.0) for the 1st layer in the orange (void) region; 6.0 (1.0) for the 2nd layer in the bule (void) region; and 3.0 for the last layer.

    .. code-block:: python

		    Np = 2 # number of patterned layers
		    Nx = 100
		    Ny = 100
		    
		    thick0 = 0.1
		    pthick = [0.2,0.3]
		    thickN = 0.4

		    ep0 = 2.
		    epN = 3.
		    
		    obj.Add_LayerUniform(thick0,ep0)
		    for i in range(Np):
		        obj.Add_LayerGrid(pthick[i],Nx,Ny)
		    obj.Add_LayerUniform(thickN,epN)

		    obj.Init_Setup()

  * Patterned layer: the 1-th layer a circular hole of radius 0.5 *L*\ :sub:`x`, and the 2-nd layer has a square hole of 0.5 *L*\ :sub:`x`
  
    .. code-block:: python

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
		    ind = np.logical_and(np.abs(x-.5)<a/2 and np.abs(y-.5)<a/2))
		    epgrid2[ind]=epbkg		    
		    
		    # combine epsilon of all layers
		    epgrid = np.concatenate((epgrid1.flatten(),epgrid2.flatten()))
		    obj.GridLayer_geteps(epgrid)

  * Incident light is *s*-polarized

    .. code-block:: python

		     planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
		     obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

		     # solve for R and T
		     R,T= obj.RT_Solve(normalize=1)

* Example 3: topology optimization of reflection of a single patterned layer, see *ex3.py* in the example folder.
