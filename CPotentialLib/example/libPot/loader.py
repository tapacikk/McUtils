import os, sys

lib_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, lib_dir)
try:
    from .libPot import *
except ImportError:
    from .setup import load
    if load() is None:
        raise
    from .libPot import *
finally:
    sys.path.pop(0)