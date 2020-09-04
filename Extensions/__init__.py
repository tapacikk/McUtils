"""
A package for managing extension modules.
The existing `ExtensionLoader` will be moving here, and will be supplemented by classes for dealing with compiled extensions
"""

from .SharedLibraryManager import *

__all__ = []
__all__ += SharedLibraryManager.__all__