from .LazyTensors import Tensor
from .Derivatives import FiniteDifferenceDerivative
from ..Coordinerds import CoordinateSet, CoordinateSystem
import numpy as np, itertools

__all__ = [
    'FunctionExpansion'
]

########################################################################################################################
#
#                                           FunctionExpansion
#
class FunctionExpansionException(Exception):
    pass
class FunctionExpansion:
    """
    A class for handling expansions of an internal coordinate potential up to 4th order
    Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian
    """

    def __init__(self,
                 derivatives,
                 transforms = None,
                 center = None,
                 ref = 0,
                 weight_coefficients = True
                 ):
        """

        :param derivatives: Derivatives of the function being expanded
        :type derivatives: Iterable[np.ndarray | Tensor]
        :param transforms: Jacobian and higher order derivatives in the coordinates
        :type transforms: Iterable[np.ndarray | Tensor] | None
        :param center: the reference point for expanding aobut
        :type center: np.ndarray | None
        :param ref: the reference point value for shifting the expansion
        :type ref: float | np.ndarray
        :param weight_coefficients: whether the derivative terms need to be weighted or not
        :type weight_coefficients: bool
        """
        self._derivs = self.FunctionDerivatives(derivatives, weight_coefficients)
        self._center = center
        self.ref = ref
        if transforms is None:
            self._transf = None
        else:
            # transformation matrices from cartesians to internals
            self._transf = self.CoordinateTransforms(transforms)

    @classmethod
    def expand_function(cls, f, point,
                        order=4,
                        basis=None,
                        function_shape=None,
                        transforms=None,
                        weight_coefficients=True,
                        **fd_options
                        ):
        """Expands a function about a point up to the given order

        :param f:
        :type f: function
        :param point:
        :type point: np.ndarray | CoordinateSet
        :param order:
        :type order: int
        :param basis:
        :type basis: None | CoordinateSystem
        :param fd_options:
        :type fd_options:
        :return:
        :rtype:
        """

        derivs = FiniteDifferenceDerivative(f, function_shape=function_shape)(point, **fd_options)
        ref = f(point)
        dts = [derivs.derivative_tensor(i) for i in range(1, order+1)]
        if transforms is None:
            if basis is not None and isinstance(point, CoordinateSet):
                transforms = [point.jacobian(basis, order=i) for i in range(order)]
            else:
                transforms = None

        return cls(dts, center=point, ref=ref, transforms=transforms, weight_coefficients=weight_coefficients)

    @property
    def tensors(self):
        return [ self.get_tensor(i) for i in range(len(self._derivs)) ]

    def get_expansions(self, coords, squeeze = True):
        """

        :param coords: Coordinates to evaluate the expansion at
        :type coords: np.ndarray | CoordinateSet
        :return:
        :rtype:
        """

        if self._center is None:
            # we assume we have a vector of points
            disp = coords
            coord_axis = 1
        else:
            if coords.shape == self._center.shape:
                coords = coords[np.newaxis]
            disp = coords - self._center
            coord_axis = coords.ndim - self._center.ndim
        expansions = []
        for i, t in enumerate(self.tensors):
            # contract the tensor by the displacements until it's completely reduced
            tensr = t #type: Tensor
            for j in range(i+1):
                if j == 1:
                    # since we have multiple configurations we "roll" the generated tensor axes
                    # so that the shared ones are all at the beginning
                    roll = np.roll(np.arange(tensr.dim), coord_axis)
                    tensr = tensr.transpose(roll)
                tensr = tensr.dot(disp, axis=(-1, coord_axis))
            contraction = tensr.array
            if squeeze:
                contraction = contraction.squeeze()
            expansions.append(contraction)

        return expansions
    def expand(self, coords, squeeze = True):
        """Returns a numerical value for the expanded coordinates

        :param coords:
        :type coords: np.ndarray
        :return:
        :rtype: float | np.ndarray
        """
        ref = self.ref
        exps = self.get_expansions(coords, squeeze = squeeze)
        return ref + sum(exps)

    def __call__(self, coords, **kw):
        return self.expand(coords, **kw)

    def get_tensor(self, i):
        """Defines the overall tensors of derivatives

        :param i:
        :type i: order of derivative tensor to provide
        :return:
        :rtype: Tensor
        """
        # simple mat-vec
        if self._transf is None:
            term = self._derivs[i]
        elif i == 0:
            d0=self._derivs[0]
            term = self._transf[0]*d0 if d0 is not 0 else 0
        elif i == 1:
            terms = []
            d0 = self._derivs[0]
            d1 = self._derivs[1]
            if d0 is not 0: # we can be a bit clever about non-existant derivatives...
                terms.append(self._transf[1]*d0)
            if d1 is not 0:
                terms.append((self._transf[0] ** 2) * d1)

            term = sum(terms) if len(terms) > 0 else 0

        elif i == 2:

            terms = []
            d0 = self._derivs[0]
            d1 = self._derivs[1]
            d2 = self._derivs[2]
            if d0 is not 0:
                terms.append(self._transf[2] * d0)
            if d1 is not 0:
                terms.append(
                    2 * ( self._transf[1] * self._transf[0] * d1 ) +
                        self._transf[0] * self._transf[1] * d1
                )
            if d2 is not 0:
                terms.append((self._transf[0] ** 3) * d2)

            term = sum(terms) if len(terms) > 0 else 0

        elif i == 3:

            terms = []
            d0 = self._derivs[0]
            d1 = self._derivs[1]
            d2 = self._derivs[2]
            d3 = self._derivs[3]

            if d0 is not 0:
                terms.append(self._transf[3]*d0)
            if d1 is not 0:
                terms.append(
                    3*self._transf[2]*self._transf[0]*d1+
                    3*(self._transf[1]**2)*d1+
                    self._transf[0]*self._transf[2]*d1
                )
            if d2 is not 0:
                terms.append(
                    2*self._transf[0]*self._transf[1]*self._transf[0]*d2+
                    3*self._transf[1]*(self._transf[0]**2)*d2+
                    (self._transf[0]**2)*self._transf[1]*d2
                )
            if d3 is not 0:
                terms.append((self._transf[0]**4)*d3)

            term = sum(terms) if len(terms) > 0 else 0

        else:
            raise FunctionExpansionException("{}: expansion only provided up to order {}".format(type(self).__name__, 4))

        return term


    class CoordinateTransforms:
        def __init__(self, transforms):
            self._transf = [Tensor.from_array(t) for t in transforms]
        def __getitem__(self, i):
            if len(self._transf) < i:
                raise FunctionExpansionException("{}: transformations requested up to order {} but only provided to order {}".format(
                    type(self).__name__,
                    i,
                    len(self._transf)
                ))
            return self._transf[i]
        def __len__(self):
            return len(self._transf)
    class FunctionDerivatives:
        def __init__(self, derivs, weight_coefficients = True):
            self.derivs = [ Tensor.from_array(t) for t in derivs ]
            if weight_coefficients:
                self.derivs = [self.weight_derivs(t, o+1) if weight_coefficients else t for o, t in enumerate(self.derivs)]
        def weight_derivs(self, t, order = None):
            """

            :param order:
            :type order: int
            :param t:
            :type t: Tensor
            :return:
            :rtype:
            """
            weighted = t
            if order is None:
                order = len(t.shape)
            if order > 1:
                s = t.shape
                weights = np.ones(s)
                all_inds = list(range(len(s)))
                for i in range(2, order+1):
                    for inds in itertools.combinations(all_inds, i):
                        # define a diagonal slice through
                        sel = tuple(slice(None, None, None) if a not in inds else np.arange(s[a]) for a in all_inds)
                        weights[sel] = 1/np.math.factorial(i)
                weighted = weighted.mul(weights)
                # print(weights, weighted.array)

            return weighted

        def __getitem__(self, i):
            if len(self.derivs) < i:
                raise FunctionExpansionException("{}: derivatives requested up to order {} but only provided to order {}".format(
                    type(self).__name__,
                    i,
                    len(self.derivs)
                ))
            return self.derivs[i]
        def __len__(self):
            return len(self.derivs)