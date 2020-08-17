"""
Provides a small data framework for wrapping up datasets into classes for access and loading
"""
from .CommonData import *
from .AtomData import *
from .ConstantsData import *
from .BondData import *
from .WavefunctionData import *

__all__ = []
from .CommonData import __all__ as _all
__all__ += _all
from .AtomData import __all__ as _all
__all__ += _all
from .ConstantsData import __all__ as _all
__all__ += _all
from .BondData import __all__ as _all
__all__ += _all
from .WavefunctionData import __all__ as _all
__all__ += _all