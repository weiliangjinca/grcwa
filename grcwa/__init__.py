"""Top-level package for grcwa."""
from .backend import backend, set_backend
from .fft_funs import Epsilon_fft,get_fft,get_ifft
from .kbloch import Lattice_Reciprocate,Lattice_getG,Lattice_SetKs
from .rcwa import obj

__author__ = """Weiliang Jin"""
__email__ = 'jwlaaa@gmail.com'
__version__ = '0.1.2'
