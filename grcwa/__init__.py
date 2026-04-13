"""Top-level package for grcwa."""

from .backend import backend, set_backend
from .fft_funs import Epsilon_fft, Epsilon_fft_pol, get_fft, get_ifft
from .kbloch import Lattice_getG, Lattice_Reciprocate, Lattice_SetKs
from .rcwa import obj

__all__ = [
    "Epsilon_fft",
    "Epsilon_fft_pol",
    "Lattice_Reciprocate",
    "Lattice_SetKs",
    "Lattice_getG",
    "backend",
    "get_fft",
    "get_ifft",
    "obj",
    "set_backend",
]

__author__ = """Weiliang Jin"""
__email__ = "jwlaaa@gmail.com"
__version__ = "0.1.3"
