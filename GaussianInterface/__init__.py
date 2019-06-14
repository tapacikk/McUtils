'''A module for hooking into Gaussian. It'd be much, much nicer to just be able to use psi4 and its native python bindings,
but... well we can't have everything.
'''
from .GaussianImporter import GaussianLogReader, GaussianFChkReader
from .GaussianJob import GaussianJob