import numpy as np
from .CoordinateSystemConverter import CoordinateSystemConverters as converters, CoordinateSystemConverter
from .CoordinateUtils import is_multiconfig, mc_safe_apply

__all__ = [
    "CoordinateSystem",
    "BaseCoordinateSystem",
    "CoordinateSystemError"
]

######################################################################################################
##
##                                   CoordinateSystem Class
##
######################################################################################################

class CoordinateSystem:
    """A representation of a coordinate system. It doesn't do much on its own but it *does* provide a way
    to unify internal, cartesian, derived type coordinates

    """
    def __init__(self,
                 name=None, basis=None, matrix=None, dimension=None,
                 jacobian_prep=None, coordinate_shape=None,
                 converter_options=None
                 ):
        """Sets up the CoordinateSystem object

        :param name: a name to give to the coordinate system
        :type name: str
        :param basis: a basis for the coordinate system
        :type basis:
        :param matrix: an expansion coefficient matrix for the set of coordinates in its basis
        :type matrix: np.ndarray | None
        :param dimension: the dimension of a single configuration in the coordinate system (for validation)
        :type dimension: Iterable[None | int]
        :param jacobian_prep: a function for preparing coordinates to be used in computing the Jacobian
        :type jacobian_prep: function | None
        :param coordinate_shape: the actual shape of a single coordinate in the coordinate system
        :type coordinate_shape: iterable[int]
        """

        if dimension is None and matrix is not None:
            dimension = (matrix.shape[-1],)
        if coordinate_shape is None:
            coordinate_shape = dimension
        if matrix is not None and (coordinate_shape is not None and coordinate_shape[-1] != matrix.shape[-1]):
            raise CoordinateSystemError(
                "{}: expansion matrix shape {} must be compatible with coordinate shape {}".format(
                    type(self).__name__,
                    matrix.shape,
                    coordinate_shape
                )
            )

        if converter_options is None:
            converter_options = {}

        self.name = name
        self._basis = basis
        self._matrix = matrix
        self._dimension = dimension
        self.jacobian_prep = jacobian_prep
        self.coordinate_shape = coordinate_shape
        self.converter_options = converter_options
        self._validate()

    def _validate(self):
        if self._matrix is None:
            pass
        elif len(self._matrix.shape) != 2:
            raise CoordinateSystemError("{}: expansion matrix must be a matrix".format(type(self).__name__))
        # elif self._matrix.shape[0] != self._matrix.shape[1]:
        #     raise CoordinateSystemException("{}: expansion matrix must square".format(type(self).__name__))

        return True

    @property
    def basis(self):
        """The basis for the representation of CoordinateSystem.matrix

        :return:
        :rtype: CoordinateSystem
        """
        return self._basis

    @property
    def matrix(self):
        """The matrix representation in the CoordinateSystem.basis
        None is shorthand for the identity matrix

        :return:
        :rtype:
        """
        return self._matrix

    @property
    def dimension(self):
        """The dimension of the coordinate system
        None means unspecified dimension

        :return:
        :rtype: int or None
        """
        return self._dimension

    def converter(self, system):
        """Gets the converter from the current system to a new system

        :param system: the target CoordinateSystem
        :type system: CoordinateSystem
        :return:
        :rtype: CoordinateSystemConverter
        """

        return converters.get_converter(self, system)

    @staticmethod
    def _apply_system_matrix(basis, coords, matrix, input_coordinate_shape, target_coordinate_shape):

        odims = target_coordinate_shape
        nones = len([a for a in odims if a is None])
        if nones > 1:
            raise CoordinateSystemError(
                "Basis {} has indeterminate shape {}. Only one None is allowed".format(
                    basis,
                    odims
                ))

        shape = coords.shape
        cdims = len(shape) - len(input_coordinate_shape)
        config_shape = shape[:cdims]
        if cdims > 0:
            new_shape = config_shape + (np.product(shape[cdims:]),)
            coords = coords.reshape(new_shape)
        # print(coords.shape, matrix.shape)
        coords = np.tensordot(coords, matrix, axes=((cdims,), (1,)))  # set up the basic expansion coordinates in the more primitive system

        # coords = coords.T
        oblock = np.product([o for o in odims if o is not None])
        out_shape = config_shape + tuple(matrix.shape[0] // oblock if o is None else o for o in odims)

        return np.reshape(coords, out_shape)  # reshape so as to actually fit the dimension of the basis

    def convert_coords(self, coords, system, **kw):
        """

        :param coords:
        :type coords: CoordinateSet
        :param system:
        :type system: CoordinateSystem
        :param kw:
        :type kw:
        :return:
        :rtype:
        """
        if system is self:
            return coords, {}
        ops = dict(system.converter_options, **self.converter_options)
        kw = dict(ops, **kw)
        if self.matrix is not None:
            coords = self._apply_system_matrix(self.basis, coords, self.matrix, self.coordinate_shape,
                                               self.basis.coordinate_shape)
            return self.basis.convert_coords(coords, system, **kw)

        elif system.matrix is not None:

            coords, convs = self.convert_coords(coords, system.basis, **kw)
            square_Q=system.matrix.shape[0] == system.matrix.shape[1]
            inv = (np.linalg.inv if square_Q else np.linalg.pinv)(system.matrix)

            coords = self._apply_system_matrix(system, coords, inv, system.basis.coordinate_shape,
                                               system.coordinate_shape)
            return coords, convs

        else:
            converter = self.converter(system)
            if is_multiconfig(coords):
                fun = lambda coords, kw=kw: converter.convert_many(coords, **kw)
            else:
                fun = lambda coords, kw=kw: converter.convert(coords, **kw)
            new_coords = mc_safe_apply(fun, coords=coords)
            return new_coords

    def displacement(self, amts):
        """Generates a displacement or matrix of displacements based on the vector or matrix amts

        :param amts:
        :type amts: np.ndarray
        :return:
        :rtype: np.ndarray
        """
        return amts # I used to think this would be relevant... but I don't really know now if it is
        # if self.matrix is None:
        #     return amts
        # else:
        #     if isinstance(amts, (float, int, np.integer, np.float)):
        #         amts = np.full((1, self.matrix.shape[-1]), amts)
        #     else:
        #         amts = np.asarray(amts)
        #         if amts.ndim == 1:
        #             amts = np.broadcast_to(np.reshape(amts, amts.shape + (1,)), amts.shape + self.matrix.shape[-1:])
        #     return np.matmul(amts, self.matrix)

    def jacobian(self,
                 coords,
                 system,
                 order = 1,
                 coordinates = None,
                 converter_options = None,
                 all_numerical=False,
                 **finite_difference_options
                 ):
        """Computes the Jacobian between the current coordinate system and a target coordinate system

        :param system: the target CoordinateSystem
        :type system: CoordinateSystem
        :param order: the order of the Jacobian to compute, 1 for a standard, 2 for the Hessian, etc.
        :type order: int
        :param coordinates: a spec of which coordinates to generate derivatives for (None means all)
        :type coordinates: None | iterable[iterable[int] | None
        :param mesh_spacing: the spacing to use when displacing
        :type mesh_spacing: float | np.ndarray
        :param prep: a function for pre-validating the generated coordinate values and grids
        :type prep: None | function
        :param fd_options: options to be passed straight through to FiniteDifferenceFunction
        :type fd_options:
        :return:
        :rtype:
        """

        from McUtils.Zachary import FiniteDifferenceDerivative

        if converter_options is None:
            converter_options = {} # convert_coords tracks the other conversion options for us

        deriv_tensor = None # default return value

        if not all_numerical:
            # sort of a hack right now: if the conversion returns 'derivs', then we use those for the FD
            # we clearly need to allow for higher derivatives, but I haven't figured out what I want to do at this stage
            ret_d_key = 'return_derivs'
            rd = converter_options[ret_d_key] if ret_d_key in converter_options else None
            converter_options[ret_d_key] = True
            test_crd, test_opts = self.convert_coords(coords, system, **converter_options)
            if rd is None:
                del converter_options[ret_d_key]
            else:
                converter_options[ret_d_key] = rd
            deriv_key = 'derivs'
            if deriv_key in test_opts and test_opts[deriv_key] is not None:
                order = order-1
                deriv_tensor = test_opts[deriv_key]
                kw = converter_options.copy()
                if ret_d_key in kw:
                    del kw[ret_d_key]
                def convert(c, s=system, dk=deriv_key, self=self, convert_kwargs=kw):
                    crds, opts = self.convert_coords(c, s, return_derivs=True, **convert_kwargs)
                    # we now have to reshape the derivatives because mc_safe_apply is only applied to the coords -_-
                    # should really make that function manage deriv shapes too, but don't know how to _tell_ it that I
                    # want it to
                    derivs = opts[dk]

                    new_deriv_shape = c.shape + derivs.shape[len(c.shape) - 1:]
                    derivs = derivs.reshape(new_deriv_shape)
                    return derivs
            else:
                convert = lambda c, s=system, kw=converter_options: self.convert_coords(c, s, **kw)[0]
        else:
            convert = lambda c, s=system, kw=converter_options: self.convert_coords(c, s, **kw)[0]
            
        if order > 0:
            self_shape = self.coordinate_shape
            if self_shape is None:
                self_shape = coords.shape[1:]
            if self_shape is None:
                raise CoordinateSystemError(
                    "{}.{}: 'coordinate_shape' {} must be tuple of ints".format(
                        type(self).__name__,
                        'jacobian',
                        self_shape
                    ))

            other_shape = system.coordinate_shape
            # if other_shape is None:
            #     raise CoordinateSystemException(
            #         "{}.{}: 'coordinate_shape' {} must be tuple of ints".format(
            #             type(self).__name__,
            #             'jacobian',
            #             other_shape
            #         ))

            for k, v in zip(
                    ('mesh_spacing', 'prep'),
                    (.001, system.jacobian_prep, None)
            ):
                if k not in finite_difference_options:
                    finite_difference_options[k] = v

            deriv = FiniteDifferenceDerivative(
                convert,
                function_shape=(self_shape, other_shape),
                **finite_difference_options
            )(coords)

            deriv_tensor = deriv.derivative_tensor(order, coordinates=coordinates)

        if deriv_tensor is None:
            raise CoordinateSystemError("derivative order '{}' less than 0".format(order))

        return deriv_tensor


######################################################################################################
##
##                                   CoordinateSystemException Class
##
######################################################################################################
class CoordinateSystemError(Exception):
    pass

######################################################################################################
##
##                                   BaseCoordinateSystem Class
##
######################################################################################################

class BaseCoordinateSystem(CoordinateSystem):
    """A CoordinateSystem object that can't be reduced further.
    A common choice might be Cartesian coordinates or internal coordinates

    """

    def __init__(self, name, dimension=None, matrix=None, coordinate_shape=None, converter_options=None):
        super().__init__(name=name,
                         dimension=dimension, basis=self, matrix=matrix, coordinate_shape=coordinate_shape,
                         converter_options=converter_options
                         )