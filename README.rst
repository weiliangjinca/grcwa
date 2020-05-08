=====
grcwa
=====
.. image:: https://img.shields.io/pypi/v/grcwa.svg
        :target: https://pypi.python.org/pypi/grcwa

.. image:: https://img.shields.io/travis/weiliangjinca/grcwa.svg
        :target: https://travis-ci.org/weiliangjinca/grcwa

.. image:: https://readthedocs.org/projects/grcwa/badge/?version=latest
        :target: https://grcwa.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


grcwa (autoGradable RCWA) is a python implementation of rigorous
coupled wave analysis (RCWA) for arbitrarily shaped photonic crystal
slabs.

* Free software: MIT license
* Documentation: https://grcwa.readthedocs.io.

Features
---------
* Each photonic crystal layer can have arbitrary dielectric profile,
  which is specified through a *2D* array whose values correspond to
  dielectric constants on structured grids.
* **autograd** is integrated into the package, allowing for automated
  and fast gradient evaluations for the sake of large-scale
  optimizations. Autogradable parameters include dielectric constant on
  every grid, frequency, angles, thickness of each layer, and
  periodicity (however the ratio of periodicity along the two lateral
  directions must be fixed).


Quick Start
-----------
* Installation:

  .. code-block:: python
		  
		  pip install grcwa

* Setup RCWA calucation
  
  ..
     .. code-block:: python

		     import grcwa
		     grcwa.set_backend('autograd') # if autograd needed


		     # lattice constants
		     L1 = [0.1,0]
		     L2 = [0,0.1]
		     # Truncation order (actual number might be smaller)
		     nG = 101
		     # frequency
		     freq = 1.
		     # angle
		     theta = 0.
		     phi = 0.

		     # setup RCWA
		     obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=1)


		     obj.Add_LayerUniform(thick0,epsuniform0)
		     obj.Add_LayerGrid(pthick[i],Nx,Ny)
		     obj.Add_LayerUniform(thickN,epsuniformN)

		     obj.Init_Setup(Pscale=Pscale,Gmethod=0)

		     planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
		     obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
		     obj.GridLayer_geteps(epgrid)
		     R,T= obj.RT_Solve(normalize=0)


Citing
-------

If you find **grcwa** useful for your research, please cite the
following paper:
::

   @article{Jin2020,
     title = {Inverse design of lightweight broadband  reflector for efficient lightsail propulsion},
     author ={Jin, Weiliang and Li, Wei and Orenstein, Meir and Fan, Shanhui},
     year = {2020},
     journal = {TODO},
   }


Acknowledgements
----------------

My implementation of RCWA received helpful discussions from `Dr. Zin
Lin
<https://scholar.google.com/citations?user=3ZgzHLYAAAAJ&hl=en>`_. Many
details of implementations were referred to a RCWA package implemented
in c called `S4 <https://github.com/victorliu/S4>`_. The idea of
integrating **Autograd** into RCWA package rather than deriving
adjoint-variable gradient by hand was inspired by a discussion with
Dr. Ian Williamson and Dr. Momchil Minkov. The backend and many other
styles follow their implementation in `legume
<https://github.com/fancompute/legume>`_, Haiwen Wang and Cheng Guo
provided useful feedback. Lastly, the template was created to
Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
