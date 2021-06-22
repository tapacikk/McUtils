"""
Provides a set of decorators that allow code to be agnostic to whether Numba exists or not
"""
import warnings, typing

__all__ = [
    'njit',
    'jit',
    'type_spec',
    'numba_decorator',
    'objmode'
]

def load_numba(warn=False):
    try:
        import numba
    except ImportError:
        if isinstance(warn, str) and warn == 'raise':
            raise
        if warn:
            warnings.warn("Numba not installed/code will be slower")
        numba = None

    return numba

def njit(*args, warn=False, **kwargs):
    return numba_decorator(*args, method='njit', warn=warn, **kwargs)

def jit(*args, warn=False, nopython=False, **kwargs):
    return numba_decorator(*args, method='jit', warn=warn, nopython=nopython, **kwargs)

def numba_decorator(*args, method=None, warn=False, **kwargs):
    numba = load_numba(warn=warn)
    if numba is not None:
        return getattr(numba, method)(*args, **kwargs)
    else:
        if len(args) > 0:
            return args[0]
        else:
            return lambda f:f

def type_spec(t, warn=False):
    numba = load_numba(warn=warn)
    if numba is not None:
        return getattr(numba, t)
    else:
        return typing.Any

class _noop_context:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def objmode(*args, warn=False, **kwargs):
    nb = load_numba(warn=warn)
    if nb is not None:
        return nb.objmode(*args, **kwargs)
    else:
        return _noop_context()