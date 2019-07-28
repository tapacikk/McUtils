'''
A module for hooking into Gaussian. It'd be much, much nicer to just be able to use psi4 and its native python bindings,
but... well we can't have everything.
'''
from .GaussianImporter import *
from .GaussianJob import *

from .GaussianImporter import __all__ as GaussianImporter__all__
from .GaussianImporter import __all__ as GaussianJob__all__
__all__ = GaussianImporter__all__ + GaussianJob__all__