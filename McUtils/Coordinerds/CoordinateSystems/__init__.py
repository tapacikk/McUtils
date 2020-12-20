"""
Provides a framework for using coordinates with explicit reference to an underlying coordinate system
"""

from .CoordinateSet import *
from .CommonCoordinateSystems import *
from .CoordinateSystem import *
from .CoordinateSystemConverter import *

__all__ = []
from .CoordinateSet import __all__ as exposed
__all__ += exposed
from .CommonCoordinateSystems import __all__ as exposed
__all__ += exposed
from .CoordinateSystem import __all__ as exposed
__all__ += exposed
from .CoordinateSystemConverter import __all__ as exposed
__all__ += exposed

CoordinateSystemConverters._preload_converters()