from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Extensions import *
from McUtils.Data import *
import sys, os, numpy as np

class ExtensionsTests(TestCase):

    @validationTest
    def test_BasicTypeSig(self):
        sig = FunctionSignature(
            "my_func",
            Argument("num_1", RealType),
            Argument("num_2", RealType, default=5),
            Argument("some_int", IntType)
        )
        self.assertEquals(sig.cpp_signature, "void my_func(double num_1, double num_2, int some_int)")
    @validationTest
    def test_SOSig(self):
        lib_file = TestManager.test_data('libmbpol.so')
        mbpol = SharedLibraryFunction(lib_file,
                                      FunctionSignature(
                                          "calcpot_",
                                          Argument("nw", PointerType(IntType)),
                                          Argument("energy", PointerType(RealType)),
                                          Argument("coords", ArrayType(RealType))
                                      )
                                      )
        self.assertTrue("SharedLibraryFunction(FunctionSignature(calcpot_(Argument('nw', PointerType(PrimitiveType(int)))" in repr(mbpol))

    @validationTest
    def test_SharedLibraryFunction(self):
        lib_file = TestManager.test_data('libmbpol.so')
        mbpol = SharedLibraryFunction(lib_file,
                                      FunctionSignature(
                                          "calcpot_",
                                          Argument("nwaters", PointerType(IntType)),
                                          Argument("energy", PointerType(RealType)),
                                          Argument("coords", ArrayType(RealType)),
                                          return_type=None,
                                          defaults={'energy':0}
                                      ),
                                      return_handler=lambda r,kw:SharedLibraryFunction.uncast(kw['energy']) / 627.5094740631
                                      )
        water = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])

        # print(mbpol(nwaters=1, coords=water))
        self.assertGreater(mbpol(nwaters=1, coords=water), .005)

        water = np.array([  # some random structure Mathematica got from who knows where...
            [-0.063259, -0.25268,    0.2621],
            [ 0.74277,   0.26059,    0.17009],
            [-0.67951,  -0.0079118, -0.43219]
        ])

        # print(mbpol(nwaters=1, coords=water))
        self.assertGreater(mbpol(nwaters=1, coords=water), .0006)

        water = np.array([  # some structure from the MBX tests...
            [-0.0044590985, -0.0513425796,  0.0000158138],
            [ 0.9861302114, -0.0745730984,  0.0000054324],
            [-0.1597470923,  0.8967180895, -0.0000164932]
        ])

        # print(mbpol(nwaters=1, coords=water))
        self.assertGreater(mbpol(nwaters=1, coords=water), .001)

    @validationTest
    def test_SharedLibrary(self):
        lib_file = TestManager.test_data('libmbpol.so')
        mbpol = SharedLibrary(
            lib_file,
            get_pot=dict(
                name='calcpot_',
                nwaters=(int,),
                energy=(float,),
                coords=[float],
                return_type=None,
                defaults={'energy': 0},
                return_handler=lambda r, kw: SharedLibraryFunction.uncast(kw['energy']) / 627.5094740631
            ),
            get_pot_grad = dict(
                name='calcpotg_',
                nwaters=(int,),
                energy=(float,),
                coords=[float],
                grad=[float],
                return_type=None,
                prep_args=lambda kw:[kw.__setitem__('grad', np.zeros(kw['nwaters']*9)), kw][1],
                defaults={'grad':None, 'energy': 0},
                return_handler=lambda r, kw: {
                    'grad':kw['grad'].reshape(-1, 3) / 627.5094740631,
                    'energy':SharedLibraryFunction.uncast(kw['energy']) / 627.5094740631
                }
            )
        )

        water = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])

        # print(mbpol(nwaters=1, coords=water))
        self.assertGreater(mbpol.get_pot_grad(nwaters=1, coords=water)['energy'], .005)
        self.assertEquals(mbpol.get_pot_grad(nwaters=1, coords=water)['grad'].shape, (3, 3))

        water = np.array([  # some random structure Mathematica got from who knows where...
            [-0.063259, -0.25268,    0.2621],
            [ 0.74277,    0.26059,   0.17009],
            [-0.67951,  -0.0079118, -0.43219]
        ])

        # print(mbpol(nwaters=1, coords=water))
        self.assertGreater(mbpol.get_pot_grad(nwaters=1, coords=water)['energy'], .0006)
        self.assertEquals(mbpol.get_pot_grad(nwaters=1, coords=water)['grad'].shape, (3, 3))

        water = np.array([  # some structure from the MBX tests...
            [-0.0044590985, -0.0513425796, 0.0000158138],
            [0.9861302114, -0.0745730984, 0.0000054324],
            [-0.1597470923, 0.8967180895, -0.0000164932]
        ])

        # print(mbpol(nwaters=1, coords=water))
        self.assertGreater(mbpol.get_pot_grad(nwaters=1, coords=water)['energy'], .001)
        self.assertEquals(mbpol.get_pot_grad(nwaters=1, coords=water)['grad'].shape, (3, 3))

    @validationTest
    def test_FFI(self):
        lib_dir = TestManager.test_data('LegacyMBPol')
        mbpol = FFIModule.from_lib(lib_dir, extra_link_args=['-mmacosx-version-min=12.0'])

        water = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])

        # print(mbpol.get_pot_grad(nwaters=1, coords=water))
        self.assertGreater(mbpol.get_pot_grad(nwaters=1, coords=water)['energy'], .005)

        water = np.array([  # some random structure Mathematica got from who knows where...
            [-0.063259, -0.25268,    0.2621],
            [ 0.74277,   0.26059,    0.17009],
            [-0.67951,  -0.0079118, -0.43219]
        ])

        # print(mbpol.get_pot_grad(nwaters=1, coords=water))
        self.assertGreater(mbpol.get_pot_grad(nwaters=1, coords=water)['energy'], .0006)

        water = np.array([  # some structure from the MBX tests...
            [-0.0044590985, -0.0513425796, 0.0000158138],
            [0.9861302114, -0.0745730984, 0.0000054324],
            [-0.1597470923, 0.8967180895, -0.0000164932]
        ])

        # print(mbpol.get_pot_grad(nwaters=1, coords=water))
        self.assertGreater(mbpol.get_pot_grad(nwaters=1, coords=water)['energy'], .001)


    @debugTest
    def test_FFI_threaded(self):
        lib_dir = TestManager.test_data('LegacyMBPol')
        mbpol = FFIModule.from_lib(lib_dir, extra_link_args=['-mmacosx-version-min=12.0']
                                   # , recompile=True
                                   )

        from Peeves.Timer import Timer
        waters = np.array([
                              [
                                  [0, 0, 0],
                                  [1, 0, 0],
                                  [0, 1, 0]
                              ],
                              [  # some random structure Mathematica got from who knows where...
                                  [-0.063259, -0.25268, 0.2621],
                                  [0.74277, 0.26059, 0.17009],
                                  [-0.67951, -0.0079118, -0.43219]
                              ],
                              [  # some structure from the MBX tests...
                                  [-0.0044590985, -0.0513425796, 0.0000158138],
                                  [0.9861302114, -0.0745730984, 0.0000054324],
                                  [-0.1597470923, 0.8967180895, -0.0000164932]
                              ]
                          ] * 3
                          )

        with Timer():
            res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
            print(np.mean(res))
        print("="*100)
        print(waters[0])
        print(mbpol.get_pot(nwaters=1, coords=waters[0]))
        with Timer():
            res = np.array([mbpol.get_pot(nwaters=1, coords=w) for w in waters])
            print(np.mean(res))


