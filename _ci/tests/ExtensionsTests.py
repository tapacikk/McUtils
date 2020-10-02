
from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Extensions import *
import sys, os, numpy as np

class ExtensionsTests(TestCase):

    @debugTest
    def test_BasicTypeSig(self):
        sig = FunctionSignature(
            "my_func",
            Argument("num_1", RealType),
            Argument("num_2", RealType, default=5),
            Argument("some_int", IntType)
        )
        self.assertEquals(sig.cpp_signature, "void my_func(double num_1, double num_2, int some_int)")
    @debugTest
    def test_SOSig(self):
        lib_file = TestManager.test_data('libmbpol.so')
        mbpol = SharedLibraryFunction(lib_file,
                              FunctionSignature(
                                  "calcpot_",
                                  Argument("nw", PointerType(IntType)),
                                  Argument("num_2", PointerType(RealType)),
                                  Argument("coords", ArrayType(RealType))
                              )
                              )
        self.assertTrue("SharedLibraryFunction(FunctionSignature(calcpot_(Argument('nw', PointerType(PrimitiveType(int)))" in repr(mbpol))
    @debugTest
    def test_SOSig(self):
        lib_file = TestManager.test_data('libmbpol.so')
        mbpol = SharedLibraryFunction(lib_file,
                                      FunctionSignature(
                                          "calcpot_",
                                          Argument("nw", PointerType(IntType)),
                                          Argument("num_2", PointerType(RealType)),
                                          Argument("coords", ArrayType(RealType))
                                      )
                                      )
        self.assertTrue(
            "SharedLibraryFunction(FunctionSignature(calcpot_(Argument('nw', PointerType(PrimitiveType(int)))" in repr(
                mbpol))



"""
import ctypes, numpy as np, os
import multiprocessing as mproc
from numpy.ctypeslib import ndpointer
​
class MBPolCaller:
    lib_loc = "/Users/Mark/Documents/UW/Research/Notebooks/Jupyter/data/"
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
​
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
​
def call_mbpol(coords):
    block_size = coords.shape[-3] * 3 * 3
    nwalkers = np.prod(coords.shape[:-3])
    carray = mproc.Array('d', coords.flatten().data)
    pool = mproc.Pool(initializer=_init, initargs=(carray, block_size))  # maybe want to reuse if possible...
    res = pool.map(_call, range(int(nwalkers)))
    return np.reshape(np.array(res), coords.shape[:-3])
​
if __name__ == "__main__":
    nwalkers = 50000
    x = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.]
    ]).T
    x = np.ascontiguousarray(np.broadcast_to(x[np.newaxis, np.newaxis], (nwalkers, 6, 3, 3)))
​
    from Peeves import Timer
​
    with Timer("Unparallelized"):
        res = np.zeros(nwalkers)
        mbpol = MBPolCaller()
        for i,crd in enumerate(x):
            res[i] = mbpol.call_single(crd)
    print("Average Energy:", np.average(res))
​
    with Timer("Parallelized"):
        res2 = call_mbpol(x)
    print("Average Energy:", np.average(res2))
    """