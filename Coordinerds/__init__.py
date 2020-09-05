"""The Coordinerds package implements stuff for dealing with coordinates and generalized coordinate systems

It provides a semi-symbolic way to represent a CoordinateSystem and a CoordinateSet that provides coordinates within a
coordinate system. A flexible system for converting between coordinate systems and manipulating coordinates will be
provided.

"""

from .CoordinateSystems import *
from .CoordinateTransformations import *
from .Conveniences import *

__all__ = []
from .CoordinateSystems import __all__ as exposed
__all__ += exposed
from .CoordinateTransformations import __all__ as exposed
__all__ += exposed