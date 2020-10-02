"""
A package for managing extension modules.
The existing `ExtensionLoader` will be moving here, and will be supplemented by classes for dealing with compiled extensions
"""

from .SharedLibraryManager import *
from .ArgumentSignature import *

__all__ = []
from .ArgumentSignature import __all__ as exposed
__all__ += exposed
from .SharedLibraryManager import __all__ as exposed
__all__ += exposed