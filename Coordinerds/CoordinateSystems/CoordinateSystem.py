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
                 name=None, basis=None, matrix=None, inverse=None,
                 dimension=None, origin=None, coordinate_shape=None,
                 jacobian_prep=None,
                 converter_options=None
                 ):
        """
        Sets up the CoordinateSystem object

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

        if dimension is None:
            if origin is not None:
                dimension = origin.shape
            elif matrix is not None:
                dimension = (matrix.shape[-1],)

        if coordinate_shape is None:
            coordinate_shape = dimension
        # if matrix is not None and (coordinate_shape is not None and coordinate_shape[-1] != matrix.shape[-1]):
        #     raise CoordinateSystemError(
        #         "{}: expansion matrix shape {} must be compatible with coordinate shape {}".format(
        #             type(self).__name__,
        #             matrix.shape,
        #             coordinate_shape
        #         )
        #     )

        if converter_options is None:
            converter_options = {}

        self.name = name
        self._basis = basis
        self._matrix = matrix
        self._inv = inverse
        self._origin = origin
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
        """
        :return: The basis for the representation of `matrix`
        :rtype: CoordinateSystem
        """
        return self._basis
    @property
    def origin(self):
        """
        :return: The origin for the expansion defined by `matrix`
        :rtype: np.ndarray
        """

        return self._origin
    @property
    def matrix(self):
        """
        The matrix representation in the `CoordinateSystem.basis`
        `None` is shorthand for the identity matrix

        :return: mat
        :rtype:  np.ndarray
        """
        return self._matrix
    @property
    def inverse(self):
        """
        The inverse of the representation in the `basis`.
        `None` is shorthand for the inverse or pseudoinverse of `matrix`.

        :return: inv
        :rtype:  np.ndarray
        """
        if self._inv is None and self._matrix is not None:
            square_Q=self.matrix.shape[0] == self.matrix.shape[1]
            self._inv = (np.linalg.inv if square_Q else np.linalg.pinv)(self.matrix)
        return self._inv
    @property
    def dimension(self):
        """
        The dimension of the coordinate system.
        `None` means unspecified dimension

        :return: dim
        :rtype: int or None
        """
        return self._dimension

    def converter(self, system):
        """
        Gets the converter from the current system to a new system

        :param system: the target CoordinateSystem
        :type system: CoordinateSystem
        :return: converter object
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
        n = matrix.shape[1]
        needs_reshape = shape[-1] != n

        if needs_reshape:
            cum_prods = np.cumprod(np.flip(shape))
            pos = np.where(cum_prods==n)[0]
            if len(pos) == 0:
                raise ValueError("Coordinate array of shape '{}' can't work with an expansion matrix of shape '{}'".format(
                    shape,
                    matrix.shape
                ))
            pos=pos[0]
            excess_shape = shape[:-(pos+1)]
            new_shape = shape[:-(pos+1)]+(n,)
            coords = np.reshape(coords, new_shape)
        else:
            pos = -1
            excess_shape = shape[:-1]

        # set up the basic expansion coordinates in the more primitive system
        coords = np.tensordot(coords, matrix, axes=((-1,), (1,)))
        # then reshape so as to actually fit the shape of the basis
        out_shape = excess_shape + target_coordinate_shape
        if nones == 1:
            the_stuff = np.product([z for z in out_shape if z is not None])
            leftover = int(np.product(coords.shape) / the_stuff)
            out_shape = tuple(z if z is not None else leftover for z in out_shape)
        coords = np.reshape(coords, out_shape)

        return coords

    def convert_coords(self, coords, system, **kw):
        """
        Converts coordiantes from the current coordinate system to _system_

        :param coords:
        :type coords: CoordinateSet
        :param system:
        :type system: CoordinateSystem
        :param kw: options to be passed through to the converter object
        :type kw:
        :return: the converted coordiantes
        :rtype: tuple(np.ndarray, dict)
        """
        if system is self:
            return coords, self.converter_options

        ops = dict(system.converter_options, **self.converter_options)
        kw = dict(ops, **kw)
        if self.matrix is not None:
            # This very commonly means that we're doing an expansion in some coordinate set,
            #   but there's an equilibrium value or 'origin' that we need to shift off...
            # For example, with normal modes we need to shift by the equilibrium value of the coordinates
            #   both when we convert _to_ normal modes (in which case the modes are `system`) and when we convert _from_
            #   normal modes (in which case the modes are `self`)
            coords = self._apply_system_matrix(self.basis, coords,
                                               self.matrix,
                                               self.coordinate_shape,
                                               self.basis.coordinate_shape
                                               )
            orig = self.origin
            if orig is not None:
                extra = coords.ndim-orig.ndim
                if extra>0:
                    orig = np.reshape(orig, (1,)*extra+orig.shape)
                coords = coords + orig
            return self.basis.convert_coords(coords, system, **kw)

        elif system.matrix is not None:
            coords, convs = self.convert_coords(coords, system.basis, **kw)
            inv = system.inverse
            orig = system.origin
            if orig is not None:
                extra = coords.ndim - orig.ndim
                if extra > 0:
                    orig = np.reshape(orig, (1,) * extra + orig.shape)
                coords = coords - orig
            coords = self._apply_system_matrix(system, coords,
                                               inv,
                                               system.basis.coordinate_shape,
                                               system.coordinate_shape
                                               )
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
        """
        Generates a displacement or matrix of displacements based on the vector or matrix amts
        The relevance of this method has become somewhat unclear...

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

    def derivatives(self,
                    coords,
                    function,
                    order=1,
                    coordinates=None,
                    result_shape=None,
                    **finite_difference_options
                    ):
        """
        Computes derivatives for an arbitrary function with respect to this coordinate system.
        Basically a more flexible version of `jacobian`.

        :param function:
        :type function:
        :param order:
        :type order:
        :param coordinates:
        :type coordinates:
        :param finite_difference_options:
        :type finite_difference_options:
        :return: derivative tensor
        :rtype: np.ndarray
        """

        from McUtils.Zachary import FiniteDifferenceDerivative

        deriv_tensor = None
        if order > 0:
            self_shape = self.coordinate_shape
            if self_shape is None:
                self_shape = coords.shape[1:]
            if self_shape is None:
                raise CoordinateSystemError(
                    "{}.{}: 'coordinate_shape' {} must be tuple of ints".format(
                        type(self).__name__,
                        'derivatives',
                        self_shape
                    ))

            # if other_shape is None:
            #     raise CoordinateSystemException(
            #         "{}.{}: 'coordinate_shape' {} must be tuple of ints".format(
            #             type(self).__name__,
            #             'jacobian',
            #             other_shape
            #         ))

            for k, v in zip(
                    ('mesh_spacing',),
                    (.001,)
            ):
                if k not in finite_difference_options:
                    finite_difference_options[k] = v

            deriv = FiniteDifferenceDerivative(
                function,
                function_shape=(self_shape, result_shape),
                **finite_difference_options
            )(coords)

            deriv_tensor = deriv.derivative_tensor(order, coordinates=coordinates)

        if deriv_tensor is None:
            raise CoordinateSystemError("derivative order {} <= 0".format(order))

        return deriv_tensor

    def jacobian(self,
                 coords,
                 system,
                 order = 1,
                 coordinates = None,
                 converter_options = None,
                 all_numerical=False,
                 **finite_difference_options
                 ):
        """
        Computes the Jacobian between the current coordinate system and a target coordinate system

        :param system: the target CoordinateSystem
        :type system: CoordinateSystem
        :param order: the order of the Jacobian to compute, 1 for a standard, 2 for the Hessian, etc.
        :type order: int | Iterable[int]
        :param coordinates: a spec of which coordinates to generate derivatives for (None means all)
        :type coordinates: None | iterable[iterable[int] | None
        :param mesh_spacing: the spacing to use when displacing
        :type mesh_spacing: float | np.ndarray
        :param prep: a function for pre-validating the generated coordinate values and grids
        :type prep: None | function
        :param fd_options: options to be passed straight through to FiniteDifferenceFunction
        :type fd_options:
        :return: derivative tensor
        :rtype: np.ndarray
        """

        # print(system)
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
                if isinstance(order, int):
                    order = order-1
                else:
                    order = [o-1 for o in order]
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

        need_derivs = max(order)>0 if not isinstance(order, int) else order > 0
        if need_derivs:
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
                    ('mesh_spacing', 'prep', 'stencil'),
                    (.001, system.jacobian_prep, 7)
            ):
                if k not in finite_difference_options:
                    finite_difference_options[k] = v

            deriv = FiniteDifferenceDerivative(
                convert,
                function_shape=(self_shape, other_shape),
                **finite_difference_options
            )(coords)

            if not isinstance(order, int):
                if 0 in order:
                    if order[0] != 0:
                        raise NotImplementedError("I don't want to mess with ordering so just put the 1 or 0 first...")
                    else:
                        ordo = order[1:]
                else:
                    ordo = order
            else:
                ordo = [order]
            derivs = deriv.derivative_tensor(ordo, coordinates=coordinates)
            if isinstance(order, int):
                deriv_tensor = derivs[0]
            elif deriv_tensor is None:
                deriv_tensor = derivs
            else:
                deriv_tensor = [deriv_tensor] + derivs # currently assuming you'd put

        if deriv_tensor is None:
            raise CoordinateSystemError("derivative order '{}' less than 0".format(order))

        return deriv_tensor

    def __repr__(self):
        """
        Provides a clean representation of a `CoordinateSystem` for printing
        :return:
        :rtype: str
        """
        return "CoordinateSystem({}, dimension={}, matrix={})".format(self.name, self.dimension, self.matrix)


######################################################################################################
##
##                                   CoordinateSystemException Class
##
######################################################################################################
class CoordinateSystemError(Exception):
    """
    An exception that happens inside a `CoordinateSystem` method
    """
    pass

######################################################################################################
##
##                                   BaseCoordinateSystem Class
##
######################################################################################################

class BaseCoordinateSystem(CoordinateSystem):
    """
    A CoordinateSystem object that can't be reduced further.
    A common choice might be Cartesian coordinates or internal coordinates.
    This allows us to define flexible `CoordinateSystem` subclasses that we _don't_ expect to be used as a base
    """

    def __init__(self, name, dimension=None, matrix=None, coordinate_shape=None, converter_options=None):
        super().__init__(name=name,
                         dimension=dimension, basis=self, matrix=matrix, coordinate_shape=coordinate_shape,
                         converter_options=converter_options
                         )