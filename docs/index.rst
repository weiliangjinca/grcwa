Welcome to grcwa's documentation!
======================================

grcwa (autoGradable RCWA) is a python implementation of rigorous
coupled wave analysis (RCWA) for arbitrarily shaped photonic crystal
slabs, supporting automatic differentation with autograd

Citing
-------

If you find **grcwa** useful for your research, please cite the
following paper:
::

   @article{Jin2020,
     title = {Inverse design of lightweight broadband reflector for relativistic lightsail propulsion},
     author ={Jin, Weiliang and Li, Wei and Orenstein, Meir and Fan, Shanhui},
     year = {2020},
     journal = {ACS Photonics},
     volume = {7},
     number = {9},
     pages = {2350--2355},
     year = {2020},
     publisher = {ACS Publications}
   }
   
Features
---------
.. image:: scheme.png

RCWA solves EM-scattering problems of stacked photonic crystal
slabs. As illustrated in the above figure, the photonic structure can
have *N* layers of different thicknesses and independent spatial
dielectric profiles. All layers are periodic in the two lateral
directions, and invariant along the vertical direction.

* Each photonic crystal layer can have arbitrary dielectric profile on
  the *2D* grids.
* **autograd** is integrated into the package, allowing for automated
  and fast gradient evaluations for the sake of large-scale
  optimizations. Autogradable parameters include dielectric constant on
  every grid, frequency, angles, thickness of each layer, and
  periodicity (however the ratio of periodicity along the two lateral
  directions must be fixed).
  
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   installation
   usage
   convention   
   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
