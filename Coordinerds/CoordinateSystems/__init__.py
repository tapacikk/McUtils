"""
Provides a framework for using coordinates with explicit reference to an underlying coordinate system
"""

from .CoordinateSet import *
from .CommonCoordinateSystems import *
from .CoordinateSystem import *
from .CoordinateSystemConverter import *

CoordinateSystemConverters._preload_converters()