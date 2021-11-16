"""
Provides utilities for setting up platform-independent parallelism
in a hopefully unobtrusive way
"""

__all__ = []
from .Parallelizers import *; from .Parallelizers import __all__ as exposed
__all__ += exposed
from .Runner import *; from .Runner import __all__ as exposed
__all__ += exposed