import gc, os, sys, argparse
test_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = test_dir
for _ in range(4):
    root_dir = os.path.dirname(root_dir)
sys.path.insert(0, root_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--recompile', dest='recompile', type=str, default='')
parser.add_argument('--debug', dest='debug', action='store_const', const=True, default=False)
parser.add_argument('--fast', dest='cmode', action='store_const', const="fast", default="quality")
parser.add_argument('--timerep', dest='time_report', action='store_const', const=True, default=False)
parser.add_argument('--novec', dest='test_unvectorized', action='store_const', const=True, default=False)
parser.add_argument('--threading', dest='threading', default='serial', type=str, nargs="?")
parser.add_argument('--iterations', dest='iterations', default=10, type=int, nargs="?")
parser.add_argument('--structs', dest='structs', default=5, type=int, nargs="?")
opts = parser.parse_args()

from McUtils.Extensions import *
from Peeves.Timer import Timer
import numpy as np, time

lib_file = os.path.join(test_dir, 'LegacyMBPol', 'libs', 'libmbpol.so')
mbpol_so = SharedLibrary(
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

# raise Exception(mbpol_so.get_pot(nwaters=1, coords=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])))

DynamicFFILibrary.configure_loader(
    threaded=True,
    extra_compile_args=(['-ftime-report'] if opts.time_report else []) + (['-O0'] if opts.cmode == 'fast' else []),
    include_dirs=['/usr/local/opt/llvm/include'],
    runtime_dirs=['/usr/local/opt/llvm/lib', '/usr/local/opt/llvm/lib/c++'],
    extra_link_args=['-mmacosx-version-min=12.0'],
    recompile=opts.recompile in {'dynamic', 'both'}
)
DynamicFFIFunctionLoader.load()
# #
#
#
# test_lib = os.path.join(
#     os.path.dirname(sys.modules['McUtils'].__file__),
#     "McUtils", "Extensions",
#     "FFI", "libs", "DynamicFFILibrary", "DynamicFFILibrary.so"
# )
# test_ffi = DynamicFFILibrary(
#     test_lib,
#     test_print = dict(
#         name='print_hi',
#         return_type='void'
#     ),
#     print_int = dict(
#         name='print_int',
#         i=int,
#         return_type='void'
#     ),
#     print_ret_int = dict(
#         name='print_ret_int',
#         i=int,
#         return_type=int
#     ),
#     print_int_ptr = dict(
#         name='print_int_ptr',
#         i=(int,),
#         return_type=int
#     ),
#     test_int_crd = dict(
#         name='print_int_crd',
#         i=(int,),
#         crd=(float,),
#         return_type='void'
#     ),
#     print_ij = dict(
#         name='print_ij',
#         i=(int,),
#         j=(int,),
#         return_type='void'
#     ),
#
#     print_coords=dict(
#         nwaters=(int,),
#         energy=[float],
#         coords=[float],
#         return_type=float,
#         prep_args=lambda kw: [
#             kw.__setitem__('energy', np.ones(len(kw['coords']) if kw['coords'].ndim > 2 else 1)),
#             kw][-1],
#         defaults={'energy': None},
#         return_handler=lambda r, kw: kw['energy']
#     )
# )

# raise Exception(
#     test_ffi.print_ret_int(
#         i=102
#         # i=np.array([1, 3], dtype='int8'),
#         # j=np.array([2, 4], dtype='int8')
#         # , threading_vars=['i', 'j']
#         , debug='excessive' if opts.debug else False
#     )
# )
#
# test_ffi.test_print(debug='excessive')

# res = test_ffi.print_coords(
#     nwaters=1,
#     coords=np.array([
#         [
#             [0, 0, 0],
#             [1, 0, 0],
#             [0, 1, 0]
#         ],
#         [
#             [0, 0, 0],
#             [1, 0, 0],
#             [0, 1, 0]
#         ],
#     ], dtype=float
#     ),
#     threading_vars=['energy', 'coords']
# )
# raise Exception(res)


# CLoader(os.path.join(test_dir, "LegacyMBPol", "libs", 'mbpol')).custom_make(
#     True, os.path.join(test_dir, "LegacyMBPol", "libs", 'mbpol')
# )

lib_file = os.path.join(test_dir, "LegacyMBPol", "libs", "libmbpol.so")
# lib_file = os.path.join(test_dir, "libmbpol.so")
mbpol_ffi = DynamicFFILibrary(
    lib_file,
    get_pot=dict(
        name='calcpot_',
        nwaters=(int,),
        energy=[float],
        coords=[float],
        return_type=float,
        prep_args=lambda kw:[
            kw.__setitem__('energy', np.zeros(len(kw['coords']) if kw['coords'].ndim > 2 else 1)),
            kw][-1],
        defaults={'energy': None},
        return_handler=lambda r, kw: kw['energy'] / 627.5094740631
    ),
    get_pot_grad = dict(
        name='calcpotg_',
        nwaters=(int,),
        energy=(float,),
        coords=[float],
        grad=[float],
        return_type=float,
        prep_args=lambda kw:[
            kw.__setitem__('grad', np.zeros(kw['coords'].shape)),
            kw.__setitem__('energy', np.zeros(len(kw['coords']) if kw['coords'].ndim > 2 else 1)),
            kw][-1],
        defaults={'grad':None, 'energy': None},
        return_handler=lambda r, kw: {
            'grad':kw['grad'] / 627.5094740631,
            'energy': kw['energy'] / 627.5094740631
        }
    )
)

# print(id(mbpol_ffi), id(mbpol_ffi.get_pot_grad))
# coords = np.array([
#     [
#         [0, 0, 0],
#         [1., 0, 0],
#         [0, 1., 0]
#     ],
#     [
#         [0, 0, 0],
#         [2., 0, 0],
#         [0, 1., 0]
#     ]
# ], dtype=float)
# raise Exception(
#     mbpol_ffi.get_pot(
#         nwaters=1,
#         coords=coords
#         , threading_vars=['energy', 'coords']
#         , threading_mode=opts.threading
#         , debug='excessive' if opts.debug else False
#     )
# )

# might need export CC=/usr/local/opt/llvm/bin/clang
lib_dir = os.path.join(test_dir, 'LegacyMBPol')
with Timer(tag="Compilation", print_times=((opts.recompile in {'ffi', 'both'}) and opts.time_report)):
    mbpol = FFIModule.from_lib(lib_dir,
                               threaded=True,
                               extra_compile_args=(['-ftime-report'] if opts.time_report else []) + (['-O0'] if opts.cmode == 'fast' else []),
                               include_dirs=['/usr/local/opt/llvm/include'],
                               runtime_dirs=['/usr/local/opt/llvm/lib', '/usr/local/opt/llvm/lib/c++'],
                               extra_link_args=['-mmacosx-version-min=12.0'],
                               recompile=opts.recompile in {'ffi', 'both'}
                               )

waters = np.array(
    [
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ],
        [  # some random structure Mathematica got from who knows where...
            [-0.063259, -0.2526800,  0.26210],
            [ 0.742770,  0.2605900,  0.17009],
            [-0.679510, -0.0079118, -0.43219]
        ],
        [  # some structure from the MBX tests...
            [-0.0044590985, -0.0513425796,  0.0000158138],
            [ 0.9861302114, -0.0745730984,  0.0000054324],
            [-0.1597470923,  0.8967180895, -0.0000164932]
        ]
    ] * opts.structs
)

test_eng = None
test_grad = None
def check_eng_grad(res):
    global test_eng, test_grad

    if test_eng is None:
        test_eng = np.mean(res['energy'])
    elif test_eng != np.mean(res['energy']):
        raise Exception("energy {} != {}".format(test_eng, np.mean(res['energy'])))

    if test_grad is None:
        test_grad = np.mean(res['grad'])
    elif test_grad != np.mean(res['grad']):
        raise Exception("grad {} != {}".format(test_grad, np.mean(res['grad'])))

test_its = opts.iterations
if opts.test_unvectorized:
    with Timer(tag="ctypes", number=test_its) as t1:
        for _ in range(test_its):
            res = [mbpol_so.get_pot_grad(nwaters=1, coords=w) for w in waters]
            res = {
                'energy':np.array([r['energy'] for r in res]),
                'grad':np.array([r['grad'] for r in res])
            }
    check_eng_grad(res)

    # mbpol.get_pot_grad(nwaters=1, coords=waters[0], debug='all' if opts.debug else False)
    with Timer(tag="FFI + python", number=test_its) as t2:
        for _ in range(test_its):
            res = [mbpol.get_pot_grad(nwaters=1, coords=w) for w in waters]
            res = {
                'energy':np.array([r['energy'] for r in res]),
                'grad':np.array([r['grad'] for r in res])
            }
    check_eng_grad(res)

    print("Relative timing: ", t2.latest/t1.latest)

with Timer(tag="FFI vectorized", number=test_its) as t3:
    for _ in range(test_its):
        res = mbpol.get_pot_grad_vec(nwaters=1, coords=waters
                                     # , debug='excessive' if opts.debug else False
                                     )
check_eng_grad(res)

if opts.test_unvectorized:
    print("Relative timing: ", t3.latest/t1.latest)

with Timer(tag="FFI threaded", number=test_its) as t4:
    for _ in range(test_its):
        res = mbpol.get_pot_grad(nwaters=1, coords=waters, threading_var='coords',
                                 threading_mode=opts.threading
                                 , debug='excessive' if opts.debug else False
                                 )
check_eng_grad(res)

print("Relative timing: ", t4.latest / t3.latest)

"""
/usr/local/opt/llvm/bin/clang -Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -arch x86_64 -g 
-I/usr/local/opt/llvm/include -I/Users/Mark/Documents/UW/Research/Development/McUtils/McUtils/Extensions/FFI/libs -I/Users/Mark/Documents/UW/Research/Development/venv/lib/python3.9/site-packages/numpy/core/include -I/Users/Mark/Documents/UW/Research/Development/venv/include -I/Library/Frameworks/Python.framework/Versions/3.9/include/python3.9 
-c LegacyMBPol.cpp -o build/temp.macosx-10.9-x86_64-cpython-39/LegacyMBPol.o -fopenmp -std=c++17 -O0
"""
"""
/usr/local/opt/llvm/bin/clang -Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -arch x86_64 -g 
-I/usr/local/opt/llvm/include -I/Users/Mark/Documents/UW/Research/Development/McUtils/McUtils/Extensions/FFI/libs -I/Users/Mark/Documents/UW/Research/Development/venv/lib/python3.9/site-packages/numpy/core/include -I/Users/Mark/Documents/UW/Research/Development/venv/include -I/Library/Frameworks/Python.framework/Versions/3.9/include/python3.9 
-c DynamicFFILibrary.cpp -o build/temp.macosx-10.9-x86_64-cpython-39/DynamicFFILibrary.o -fopenmp -std=c++17 -O0
"""

with Timer(tag="libffi vectorized", number=test_its) as t6:
    for _ in range(test_its):
        res = mbpol_ffi.get_pot_grad(nwaters=1, coords=waters, threading_vars=['coords', 'energy', 'grad'],
                                 threading_mode=opts.threading
                                     # , debug='excessive' if opts.debug else False
                                     )
check_eng_grad(res)

print("Relative timing: ", t6.latest / t3.latest)


# # Either I fucked this up or there's some crazy RVO b.c. this is barely faster
# # than `get_pot_grad_vec` despite the fact that `get_pot_grad_vec` *should*
# # be allocating a big array (on the stack) and then _copying_ that into a `numpy` array
# with Timer(tag="buffered", number=test_its) as t5:
#     res = {
#         'energy':np.zeros(len(waters), dtype='float64'),
#         'grad':np.zeros((len(waters), 3, 3), dtype='float64')
#     }
#     for _ in range(test_its):
#         mbpol.get_pot_grad_vec_buffered(nwaters=1, coords=waters, energies=res['energy'], gradients=res['grad'])
# # print(np.mean(res['energy']))
# # print(np.mean(res['grad']))
#
# print("Relative timing: ", t5.latest/t3.latest)
# n = 0
# for _ in range(10):
#     for _ in range(10):
#         res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
#     n += np.mean(res)
#     for _ in range(10):
#         res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
#     n += np.mean(res)
#     for _ in range(10):
#         res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
#     n += np.mean(res)
#     for _ in range(10):
#         res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
#     n += np.mean(res)
#     for _ in range(10):
#         res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
#     n += np.mean(res)
#     for _ in range(10):
#         res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
#     n += np.mean(res)
#     for _ in range(10):
#         res = mbpol.get_pot(nwaters=1, coords=waters, threading_var='coords')
#     n += np.mean(res)
