"""
Defines a SharedLibrary object that makes it relatively straightforward to use
extensions direclty from .so files
"""

import os, ctypes, multiprocessing as mproc, numpy.ctypeslib as npctypes

__all__ = [
    "Argument",
    "ArgumentSignature",
    "SharedLibraryFunction"
]

# TODO: need to finish up with the actual type inference & make it
#   so that we can use a `Argument.from_value` constructor
#   Also need to have call_single do the appropriate restructing of its Arguments
#   list so that stuff can cleanly go into the C-level + add some docstring-support
class Argument:
    """
    Defines a single Argument for a C-level caller to support default values, etc.
    """
    type_map = {
        float: ctypes.c_double,
        int: ctypes.c_int,
        str: ctypes.c_char_p,  # pointer to char array
        # object: ctypes.,
        'float': ctypes.c_float,
        'double': ctypes.c_double,
        'float64': npctypes._get_scalar_type_map()
    }
    type_map.update({k.name: v for k, v in npctypes._get_scalar_type_map().items()})
    type_map.update(npctypes._get_scalar_type_map())
    inverse_type_map = {v: k.name for k, v in npctypes._get_scalar_type_map().items()}

    def __init__(self, name, dtype, default=None, pointer_type=None, array_type=None):
        """
        :param name: the name of the argument
        :type name: str
        :param dtype: the type of the argument
        :type dtype:
        :param default: the default value for the argument
        :type default:
        """
        self.name = name
        self.dtype, ptr_type, array_type = self.infer_type(dtype)
        if pointer_type is None:
            pointer_type = ptr_type
        self.pointer_type = pointer_type
        self.default = self.prep_argument(default)

    @classmethod
    def infer_type(cls, arg):
        """
        infers the type of an argument
        :param arg:
        :type arg: str | type | ctypes type
        :return:
        :rtype:
        """
        arg_type = None
        pointer = None
        array = None
        if arg in cls.type_map:
            arg_type = cls.type_map[arg]
        elif isinstance(arg, str):
            # we assume we can infer the array structure
            if arg.startswith("*"):
                pointer = True
                arg_type, ptr_type = cls.infer_type(arg_type[1:])
            elif arg.endswith("]"): # array type, but we don't necessarily need to care about the internal shape
                pointer = True
                arg_type, ptr_type = "]".join(arg.split("["))
        else:
            # we assume it's a valid type for python work
            ...

        if pointer:
            arg_type = ctypes.POINTER(arg_type)
        return arg_type, pointer

    @classmethod
    def infer_array_type(cls, argstr):
        ...

    @classmethod
    def inferred_type_string(cls, arg):
        """
        returns a type string for the inferred type
        """
        ...


class ArgumentSignature:
    """
    Defines an argument signature for a C-level caller
    """

    def __init__(self, *args, return_type=None):
        """
        :param args: the set of argument types to be passed into the caller
        :type args: Iterable[tuple]
        """
        if return_type is not None:
            return_type = Argument.infer_type(return_type)
        self._ret_type = return_type
        self._arguments = tuple(self.build_argument(x, i) for i,x in enumerate(args))
    def build_argument(self, argtup, which=None):
        """
        Converts an argument tuple into an Argument object
        :param argtup:
        :type argtup:
        :return:
        :rtype:
        """
        if isinstance(argtup, Argument):
            return argtup
        elif isinstance(argtup, str):
            argtup = ('_argument_{}'.format(which), argtup)
        elif isinstance(argtup, dict):
            if 'default' not in argtup:
                argtup['default'] = None
            argtup = (argtup['name'], argtup['dtype'], argtup['default'])
        if len(argtup) == 2:
            name, dtype = argtup
            default = None
        else:
            name, dtype, default = argtup
        return Argument(name=name, dtype=dtype, default=default)

    @property
    def args(self):
        return self._arguments

    @property
    def return_type(self):
        return self._ret_type


class SharedLibraryFunction:
    """
    An object that provides a way to call into a shared library function
    """

    def __init__(self, shared_library, function, signature,
                 call_directory=None,
                 docstring=None
                 ):
        """
        :param shared_library: the path to the shared library file you want to use
        :type shared_library: str |
        :param function_name: the name of the function
        :type function_name: str | callable
        :param call_signature: the call signature of the function
        :type call_signature: ArgumentSignature
        :param call_directory: the directory for calling
        :type call_directory: str
        :param docstring: the docstring for the function
        :type docstring: str
        """
        self._lib = None
        self._lib_file = shared_library
        self._fname = function
        self._fun = None
        self._sig = signature
        if call_directory is None:
            call_directory = os.path.dirname(shared_library)
        self._dir = call_directory  # we could be kinder here and do stuff like add support for ".." and friends

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

    def initialize(self):
        if self._fun is None:
            if isinstance(self._fname, str):  # means we need to load it from the shared lib
                self._fun = getattr(self.lib, self._fname)
            # now initialize the arg signature
            self._fun.restype = self._sig.return_type
            self._fun.argtypes = self._sig.arg_list

    def call_single(self, **kwargs):
        """
        Calls the function on a single set of coordinates
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
