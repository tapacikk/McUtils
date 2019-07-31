"""
A plotting framework that builds off of matplotlib, but potentially could use a different backend
"""

from .Graphics import *
from .Plots import *
from .Primitives import *
from .Interactive import *

from .Graphics import __all__ as Graphics__all__
from .Plots import __all__ as Plots__all__
from .Primitives import __all__ as Primitives__all__
from .Interactive import __all__ as Interactive__all__

__all__ = (
        Graphics__all__ +
        Plots__all__ +
        Primitives__all__ +
        Interactive__all__
)
