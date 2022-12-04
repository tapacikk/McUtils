"""
Defines a SharedLibrary object that makes it relatively straightforward to use
extensions direclty from .so files
"""

import os, ctypes
from .ArgumentSignature import FunctionSignature

__all__ = [
    "SharedLibraryFunction"
]

class SharedLibraryFunction:
    """
    An object that provides a way to call into a shared library function
    """

    def __init__(self,
                 shared_library,
                 signature:FunctionSignature,
                 defaults=None,
                 docstring=None,
                 call_directory=None,
                 return_handler=None
                 ):
        """
        :param shared_library: the path to the shared library file you want to use
        :type shared_library: str |
        :param function_signature: the signature of the function to load
        :type function_signature: FunctionSignature
        :param call_directory: the directory for calling
        :type call_directory: str
        :param docstring: the docstring for the function
        :type docstring: str
        """
        self._lib = None
        self._lib_file = shared_library
        self._fun = None
        self._fname = None
        self._sig = signature
        self._doc = docstring
        if call_directory is None:
            call_directory = os.path.dirname(shared_library)
        self._dir = call_directory  # we could be kinder here and do stuff like add support for ".." and friends
        self.defaults = defaults
        if return_handler is None:
            return_handler = self._manage_return
        self.return_handler = return_handler

    class InDir:
        """
        A super simple context manager that manages going into a directory and then leaving when finished
        """

        def __init__(self, dir_name):
            self._to = dir_name
            self._from = None

        def __enter__(self):
            self._from = os.getcwd()
            os.chdir(self._to)

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._from is not None:
                os.chdir(self._from)
            self._from = None

    @property
    def lib(self):
        if self._lib is None:
            with self.InDir(self._dir):
                self._lib = ctypes.cdll.LoadLibrary(self._lib_file)
        return self._lib

    @property
    def function(self):
        self.initialize()
        return self._fun
    def initialize(self):
        if self._fun is None:
            # means we need to load it from the shared lib
            if self._fname is None:
                self._fname = self._sig.name
            self._fun = getattr(self.lib, self._fname)

            # now initialize the arg signature
            self._fun.restype = self._sig.return_type
            self._fun.argtypes = self._sig.arg_types # need to figure out what type I need...
    def doc(self):
        return self._sig.cpp_signature+"\n"+self._doc
    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._sig,
            self.lib
        )

    @classmethod
    def _manage_return(cls, res, args):
        return cls.uncast(res)
    @classmethod
    def uncast(cls, res):
        if hasattr(res, '_obj'): #byref
            res = res._obj
        if hasattr(res, 'value'):
            res = res.value
        return res

    def call(self, *args, **kwargs):
        """
        Calls the function we loaded.
        This will be parallelized out to handle more complicated usages.

        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        args = self._sig.prep_args(args, kwargs, defaults=self.defaults)
        res = self.function(*args)
        args = dict(zip((a.name for a in self._sig.args), args))
        return self.return_handler(res, args)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)



if __name__ == 'nope':
    import ctypes, numpy as np, os
    import multiprocessing as mproc
    from numpy.ctypeslib import ndpointer
    class MBPolCaller:
        lib_loc = "~/Documents/UW/Research/Notebooks/Jupyter/data/"
        def __init__(self):
            self._nw = None
            self._cnw = None
            self._v = np.zeros(1)
            self._lib = None
            self._fun = None
        def initialize(self, nw):
            if self._lib is None:
                cur_dir = os.getcwd()
                try:
                    os.chdir(self.lib_loc)
                    self._lib = ctypes.cdll.LoadLibrary("./libmbpol.so")
                    self._fun = self._lib.calcpot_
                finally:
                    os.chdir(cur_dir)
                self._fun.restype = None
                self._fun.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
                ]
            if self._cnw is None: # just an optimization
                self._nw = nw
                self._cnw = ctypes.byref(ctypes.c_int32(self._nw))
        def call_single(self, coords):
            if isinstance(coords, np.ndarray):
                self.initialize(coords.shape[0])
            else:
                nw = len(coords) // 9 # N_w X 3 X 3
                self.initialize(nw)
                coords = np.array(coords)
            coords=np.ascontiguousarray(coords.astype('float64'))
            self._fun(self._cnw, self._v, coords)
            return self._v[0]
        def __call__(self, coords):
            return self.call_single(coords)

    _coords = None
    _mbpol = None
    _block_size = None
    def _init(crds, block_size):
        global _coords, _mbpol, _block_size
        _coords = crds
        _block_size = block_size
        _mbpol = MBPolCaller()
    def _call(i):
        block_start = _block_size*i
        return _mbpol.call_single(_coords[block_start:block_start+_block_size])
    def call_mbpol(coords):
        block_size = coords.shape[-3] * 3 * 3
        nwalkers = np.prod(coords.shape[:-3])
        carray = mproc.Array('d', coords.flatten().data)
        pool = mproc.Pool(initializer=_init, initargs=(carray, block_size))  # maybe want to reuse if possible...
        res = pool.map(_call, range(int(nwalkers)))
        return np.reshape(np.array(res), coords.shape[:-3])

    if __name__ == "__main__":
        nwalkers = 50000
        x = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.]
        ]).T
        x = np.ascontiguousarray(np.broadcast_to(x[np.newaxis, np.newaxis], (nwalkers, 6, 3, 3)))

        from Peeves import Timer
        with Timer("Unparallelized"):
            res = np.zeros(nwalkers)
            mbpol = MBPolCaller()
            for i,crd in enumerate(x):
                res[i] = mbpol.call_single(crd)
        print("Average Energy:", np.average(res))
        with Timer("Parallelized"):
            res2 = call_mbpol(x)
        print("Average Energy:", np.average(res2))