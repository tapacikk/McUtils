"""
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class
"""

import numpy as np, abc, enum
import scipy.interpolate as interpolate
import scipy.spatial as spat
from .Mesh import Mesh, MeshType

__all__ = [
    "Interpolator",
    "Extrapolator",
    "ProductGridInterpolator",
    "UnstructuredGridInterpolator"
]


class InterpolatorException(Exception):
    pass


######################################################################################################
##
##                                   Interpolator Class
##
######################################################################################################

class BasicInterpolator(metaclass=abc.ABCMeta):
    """
    Defines the abstract interface we'll need interpolator instances to satisfy so that we can use
    `Interpolator` as a calling layer
    """

    @abc.abstractmethod
    def __init__(self, grid, values, **opts):
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def __call__(self, points, **kwargs):
        raise NotImplementedError("abstract interface")

    @abc.abstractmethod
    def derivative(self, order):
        """
        Constructs the derivatives of the interpolator at the given order
        :param order:
        :type order:
        :return:
        :rtype: BasicInterpolator
        """
        raise NotImplementedError("abstract interface")

class ProductGridInterpolator(BasicInterpolator):
    """
    A set of interpolators that support interpolation
    on a regular (tensor product) grid
    """

    def __init__(self, grids, vals, caller=None, order=None, extrapolate=True):
        """

        :param grids:
        :type grids:
        :param points:
        :type points:
        :param caller:
        :type caller:
        :param order:
        :type order: int | Iterable[int]
        """

        if order is None:
            order = 3

        self.grids = grids
        self.vals = vals
        if caller is None:
            if isinstance(grids[0], (int, float, np.integer, np.floating)):
                grids = [grids]
            ndim = len(grids)
            if ndim == 1:
                opts = {}
                if order is not None:
                    opts["k"] = order

                caller = interpolate.PPoly.from_spline(interpolate.splrep(grids[0], vals, k=order),
                                                      extrapolate=extrapolate
                                                      )
            else:
                caller = self.construct_ndspline(grids, vals, order, extrapolate=extrapolate)
        self.caller = caller

    @classmethod
    def construct_ndspline(cls, grids, vals, order, extrapolate=True):
        """
        Builds a tensor product ndspline by constructing a product of 1D splines

        :param grids: grids for each dimension independently
        :type grids: Iterable[np.ndarray]
        :param vals:
        :type vals: np.ndarray
        :param order:
        :type order: int | Iterable[int]
        :return:
        :rtype: interpolate.NdPPoly
        """

        # inspired by `csaps` python package
        # idea is to build a spline approximation in
        # every direction (where the return values are multidimensional)
        ndim = len(grids)
        if isinstance(order, (int, np.integer)):
            order = [order] * ndim

        coeffs = vals
        x = [None]*ndim
        for i, (g, o) in enumerate(zip(grids, order)):
            og_shape = coeffs.shape
            coeffs = coeffs.reshape((len(g), -1)).T
            sub_coeffs = [np.empty(0)]*len(coeffs)
            for e,v in enumerate(coeffs):
                ppoly = interpolate.PPoly.from_spline(interpolate.splrep(g, v, k=o))
                x[i] = ppoly.x
                sub_coeffs[e] = ppoly.c
            coeffs = np.array(sub_coeffs)
            coeffs = coeffs.reshape(
                og_shape[1:]+
                    sub_coeffs[0].shape
            )
            tot_dim = ndim+i+1
            coeffs = np.moveaxis(coeffs, tot_dim-2, tot_dim-2-i)

        return interpolate.NdPPoly(coeffs, x, extrapolate=extrapolate)

    def __call__(self, *args, **kwargs):
        """
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype: np.ndarray
        """
        return self.caller(*args, **kwargs)

    def derivative(self, order):
        """
        :param order:
        :type order:
        :return:
        :rtype: ProductGridInterpolator
        """
        # ndim = len(self.grids)
        # if ndim == 1:
        return type(self)(
            self.grids,
            self.vals,
            caller=self.caller.derivative(order)
        )
        # elif ndim == 2:
        #     def caller(coords, _og=self.caller, **kwargs):
        #         return _og(coords, dx=order[0], dy=order[1], **kwargs)
        #     return type(self)(
        #         self.grids,
        #         self.vals,
        #         caller=caller
        #     )
        # else:
        #     derivs = self.caller.derivative(order)
        #     raise NotImplementedError("woof")

class UnstructuredGridInterpolator(BasicInterpolator):
    """
    Defines an interpolator appropriate for totally unstructured grids by
    delegating to the scipy `RBF` interpolators
    """

    default_neighbors=25
    def __init__(self, grid, values, order=None, neighbors=None, extrapolate=True, **opts):
        """
        :param grid:
        :type grid: np.ndarray
        :param values:
        :type values:  np.ndarray
        :param order:
        :type order: int
        :param neighbors:
        :type neighbors: int
        :param extrapolate:
        :type extrapolate: bool
        :param opts:
        :type opts:
        """
        self.extrapolate=extrapolate
        self._hull = None
        self._grid = grid

        if neighbors is None:
            neighbors = np.min([self.default_neighbors, len(grid)])

        if order is not None:
            if isinstance(order, int):
                if order == 1:
                    order = "linear"
                elif order == 3:
                    order = "cubic"
                elif order == 5:
                    order = "quintic"
                else:
                    raise InterpolatorException("{} doesn't support interpolation order '{}'".format(
                        interpolate.RBFInterpolator,
                        order
                    ))
            self.caller = interpolate.RBFInterpolator(grid, values, kernel=order, neighbors=neighbors, **opts)
        else:
            self.caller = interpolate.RBFInterpolator(grid, values, neighbors=neighbors, **opts)

    def _member_q(self, points):
        """
        Checks if the points are in the interpolated convex hull
        in the case that we aren't extrpolating so we can return
        NaN for those points
        :param points:
        :type points:
        :return:
        :rtype:
        """
        if self._hull is None:
            self._hull = spat.ConvexHull(self._grid)
            self._hull = spat.Delaunay(self._hull)
        return self._hull.find_simplex(points) >= 0
    def __call__(self, points):
        if self.extrapolate:
            return self.caller(points)
        else:
            hull_points = self._member_q(points)
            res = np.full(len(points), np.nan)
            res[hull_points] = self.caller(points[hull_points])
            return res

    def derivative(self, order):
        """
        Constructs the derivatives of the interpolator at the given order
        :param order:
        :type order:
        :return:
        :rtype: UnstructuredGridInterpolator
        """
        raise NotImplementedError("derivatives not implemented for unstructured grids")

class MultiCenteredRBF(interpolate.RBFInterpolator):

    def __init__(self, y, d,
                 parameterized_kernel:callable,
                 neighbors=5,
                 smoothing=0.0,
                 degree=None):
        if neighbors == None:
            raise NotImplementedError("not implemented for infinite neighbors")
        super().__init__(y, d,
                         kernel='linear',
                         epsilon=1.0,
                         smoothing=smoothing,
                         degree=degree
                         )
        self.kernel = parameterized_kernel

    @staticmethod
    def kernel_vector(x, y, kernel_func, out):
        """Evaluate RBFs, with centers at `y`, at the point `x`."""
        for i in range(y.shape[0]):
            out[i] = kernel_func(np.linalg.norm(x - y[i]))

    @staticmethod
    def polynomial_vector(x, powers, out):
        """Evaluate monomials, with exponents from `powers`, at the point `x`."""
        for i in range(powers.shape[0]):
            out[i] = np.prod(x ** powers[i])

    @staticmethod
    def kernel_matrix(x, kernel_func, out):
        """Evaluate RBFs, with centers at `x`, at `x`."""
        for i in range(x.shape[0]):
            for j in range(i + 1):
                out[i, j] = kernel_func(np.linalg.norm(x[i] - x[j]))
                out[j, i] = out[i, j]

    @staticmethod
    def polynomial_matrix(x, powers, out):
        """Evaluate monomials, with exponents from `powers`, at `x`."""
        for i in range(x.shape[0]):
            for j in range(powers.shape[0]):
                out[i, j] = np.prod(x[i] ** powers[j])

    @classmethod
    def _kernel_matrix(cls, x, kernel_func):
        """Return RBFs, with centers at `x`, evaluated at `x`."""
        out = np.empty((x.shape[0], x.shape[0]), dtype=float)
        cls.kernel_matrix(x, kernel_func, out)
        return out

    @classmethod
    def _polynomial_matrix(cls, x, powers):
        """Return monomials, with exponents from `powers`, evaluated at `x`."""
        out = np.empty((x.shape[0], powers.shape[0]), dtype=float)
        cls.polynomial_matrix(x, powers, out)
        return out

    # custom RBF evaluation to use a different basis function at each
    # mesh point
    @classmethod
    def _build_system(self, y, d, smoothing, kernel_func, epsilon, powers):
        """Build the system used to solve for the RBF interpolant coefficients.
        """
        p = d.shape[0]
        s = d.shape[1]
        r = powers.shape[0]

        # Shift and scale the polynomial domain to be between -1 and 1
        mins = np.min(y, axis=0)
        maxs = np.max(y, axis=0)
        shift = (maxs + mins) / 2
        scale = (maxs - mins) / 2
        # The scale may be zero if there is a single point or all the points have
        # the same value for some dimension. Avoid division by zero by replacing
        # zeros with ones.
        scale[scale == 0.0] = 1.0

        yeps = y * epsilon
        yhat = (y - shift) / scale

        # Transpose to make the array fortran contiguous. This is required for
        # dgesv to not make a copy of lhs.
        lhs = np.empty((p + r, p + r), dtype=float).T
        self.kernel_matrix(yeps, kernel_func, lhs[:p, :p])
        self.polynomial_matrix(yhat, powers, lhs[:p, p:])
        lhs[p:, :p] = lhs[:p, p:].T
        lhs[p:, p:] = 0.0
        for i in range(p):
            lhs[i, i] += smoothing[i]

        # Transpose to make the array fortran contiguous.
        rhs = np.empty((s, p + r), dtype=float).T
        rhs[:p] = d
        rhs[p:] = 0.0

        return lhs, rhs, shift, scale

    def _evaluate(self, x, y, kernel_func, epsilon, powers, shift, scale, coeffs):
        """Evaluate the RBF interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) float ndarray
            Evaluation point coordinates.
        y : (P, N) float ndarray
            Data point coordinates.
        kernel : callable
            The kernel to evaluate
        epsilon : float
            Shape parameter.
        powers : (R, N) int ndarray
            The exponents for each monomial in the polynomial.
        shift : (N,) float ndarray
            Shifts the polynomial domain for numerical stability.
        scale : (N,) float ndarray
            Scales the polynomial domain for numerical stability.
        coeffs : (P + R, S) float ndarray
            Coefficients for each RBF and monomial.

        Returns
        -------
        (Q, S) float ndarray

        """
        q = x.shape[0]
        p = y.shape[0]
        r = powers.shape[0]
        s = coeffs.shape[1]

        yeps = y * epsilon
        xeps = x * epsilon
        xhat = (x - shift) / scale

        out = np.zeros((q, s), dtype=float)
        vec = np.empty((p + r,), dtype=float)
        for i in range(q):
            self.kernel_vector(xeps[i], yeps, kernel_func, vec[:p])
            self.polynomial_vector(xhat[i], powers, vec[p:])
            # Compute the dot product between coeffs and vec. Do not use np.dot
            # because that introduces build complications with BLAS (see
            # https://github.com/serge-sans-paille/pythran/issues/1346)
            for j in range(s):
                for k in range(p + r):
                    out[i, j] += coeffs[k, j] * vec[k]

        return out

    def __call__(self, x):
        """
        Evaluate the interpolant at `x`.
        A minimal edit on the basic `RBFInterpolator.__call__` scheme

        Parameters
        ----------
        x : (Q, N) array_like
            Evaluation point coordinates.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.

        """


        x = np.asarray(x, dtype=float, order="C")
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional array.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                "Expected the second axis of `x` to have length "
                f"{self.y.shape[1]}."
                )

        if self.neighbors is None:
            out = self._evaluate(
                x, self.y, self.kernel, self.epsilon, self.powers, self._shift,
                self._scale, self._coeffs
                )
        else:
            # Get the indices of the k nearest observation points to each
            # evaluation point.
            _, yindices = self._tree.query(x, self.neighbors)
            if self.neighbors == 1:
                # `KDTree` squeezes the output when neighbors=1.
                yindices = yindices[:, None]

            # Multiple evaluation points may have the same neighborhood of
            # observation points. Make the neighborhoods unique so that we only
            # compute the interpolation coefficients once for each
            # neighborhood.
            yindices = np.sort(yindices, axis=1)
            yindices, inv = np.unique(yindices, return_inverse=True, axis=0)
            # `inv` tells us which neighborhood will be used by each evaluation
            # point. Now we find which evaluation points will be using each
            # neighborhood.
            xindices = [[] for _ in range(len(yindices))]
            for i, j in enumerate(inv):
                xindices[j].append(i)

            out = np.empty((nx, self.d.shape[1]), dtype=float)
            for xidx, yidx in zip(xindices, yindices):
                # `yidx` are the indices of the observations in this
                # neighborhood. `xidx` are the indices of the evaluation points
                # that are using this neighborhood.
                xnbr = x[xidx]
                ynbr = self.y[yidx]
                dnbr = self.d[yidx]
                snbr = self.smoothing[yidx]
                shift, scale, coeffs = self._build_and_solve_system(
                    ynbr, dnbr, snbr, self.kernel, self.epsilon, self.powers,
                    )

                out[xidx] = self._evaluate(
                    xnbr, ynbr, self.kernel, self.epsilon, self.powers, shift,
                    scale, coeffs
                    )

        out = out.view(self.d_dtype)
        out = out.reshape((nx,) + self.d_shape)
        return out

class ExtrapolatorType(enum.Enum):
    Default='Automatic'
    Error='Raise'

class Interpolator:
    """
    A general purpose that takes your data and just interpolates it without whining or making you do a pile of extra work
    """
    DefaultExtrapolator = ExtrapolatorType.Default
    def __init__(self,
                 grid,
                 vals,
                 interpolation_function=None,
                 interpolation_order=None,
                 extrapolator=None,
                 extrapolation_order=None,
                 **interpolation_opts
                 ):
        """
        :param grid: an unstructured grid of points **or** a structured grid of points **or** a 1D array
        :type grid: np.ndarray
        :param vals: the values at the grid points
        :type vals: np.ndarray
        :param interpolation_function: the basic function to be used to handle the raw interpolation
        :type interpolation_function: None | BasicInterpolator
        :param interpolation_order: the order of extrapolation to use (when applicable)
        :type interpolation_order: int | str | None
        :param extrapolator: the extrapolator to use for data points not on the grid
        :type extrapolator: Extrapolator | None | str | function
        :param extrapolation_order: the order of extrapolation to use by default
        :type extrapolation_order: int | str | None
        :param interpolation_opts: the options to be fed into the interpolating_function
        :type interpolation_opts:
        """
        self.grid = grid = Mesh(grid) if not isinstance(grid, Mesh) else grid
        self.vals = vals
        if interpolation_function is None:
            interpolation_function = self.get_interpolator(grid, vals,
                                                           interpolation_order=interpolation_order,
                                                           allow_extrapolation=extrapolator is None,
                                                           **interpolation_opts
                                                           )
        self.interpolator = interpolation_function

        if extrapolator is not None:
            if isinstance(extrapolator, ExtrapolatorType):
                if extrapolator == ExtrapolatorType.Default:
                    extrapolator = self.get_extrapolator(grid, vals, extrapolation_order=extrapolation_order)
                else:
                    raise ValueError("don't know what do with extrapolator type {}".format(extrapolator))
            elif not isinstance(extrapolator, Extrapolator):
                extrapolator = Extrapolator(extrapolator)
        self.extrapolator = extrapolator

    @classmethod
    def get_interpolator(cls, grid, vals, interpolation_order=None, allow_extrapolation=True, **opts):
        """Returns a function that can be called on grid points to interpolate them

        :param grid:
        :type grid: Mesh
        :param vals:
        :type vals: np.ndarray
        :param interpolation_order:
        :type interpolation_order: int | str | None
        :param opts:
        :type opts:
        :return: interpolator
        :rtype: function
        """
        if grid.ndim == 1:
            interpolator = ProductGridInterpolator(
                grid,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif (
                grid.mesh_type == MeshType.Structured
                or grid.mesh_type == MeshType.Regular
        ):
            interpolator = ProductGridInterpolator(
                grid.subgrids,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif grid.mesh_type == MeshType.Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            interpolator = UnstructuredGridInterpolator(
                grid,
                vals,
                order=interpolation_order,
                extrapolate=allow_extrapolation
            )
        elif grid.mesh_type == MeshType.SemiStructured:
            raise NotImplementedError("don't know what I want to do with semistructured meshes anymore")
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return interpolator

    @classmethod
    def get_extrapolator(cls, grid, vals, extrapolation_order=1, **opts):
        """
        Returns an Extrapolator that can be called on grid points to extrapolate them

        :param grid:
        :type grid: Mesh
        :param extrapolation_order:
        :type extrapolation_order: int
        :return: extrapolator
        :rtype: Extrapolator
        """

        # Extrapolator(
        #     cls(
        #         grid,
        #         vals,
        #         interpolation_order=extrapolation_order,
        #         extrapolator=Extrapolator(lambda g: np.full(g.shape, np.nan))
        #     )
        # )

        if grid.ndim == 1:
            extrapolator = ProductGridInterpolator(
                grid,
                vals,
                order=extrapolation_order,
                extrapolate=True
            )
        elif (
                grid.mesh_type == MeshType.Structured
                or grid.mesh_type == MeshType.Regular
        ):
            extrapolator = ProductGridInterpolator(
                grid.subgrids,
                vals,
                order=extrapolation_order,
                extrapolate=True
            )
        elif grid.mesh_type == MeshType.Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            extrapolator = UnstructuredGridInterpolator(
                grid,
                vals,
                neighbors=extrapolation_order+1,
                order=extrapolation_order,
                extrapolate=True
            )
        elif grid.mesh_type == MeshType.SemiStructured:
            raise NotImplementedError("don't know what I want to do with semistructured meshes anymore")
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return Extrapolator(
            extrapolator,
            **opts
        )

    def apply(self, grid_points, **opts):
        """Interpolates then extrapolates the function at the grid_points

        :param grid_points:
        :type grid_points:
        :return:
        :rtype:
        """
        # determining what points are "inside" an interpolator region is quite tough
        # instead it is probably better to allow the basic thing to interpolate and do its thing
        # and then allow the extrapolator to post-process that result
        vals = self.interpolator(grid_points, **opts)
        if self.extrapolator is not None:
            vals = self.extrapolator(grid_points, vals)
        return vals

    def derivative(self, order):
        """
        Returns a new function representing the requested derivative
        of the current interpolator

        :param order:
        :type order:
        :return:
        :rtype:
        """
        return type(self)(
            self.grid,
            self.vals,
            interpolation_function=self.interpolator.derivative(order),
            extrapolator=self.extrapolator.derivative(order) if self.extrapolator is not None else self.extrapolator
        )

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

######################################################################################################
##
##                                   Extrapolator Class
##
######################################################################################################
class Extrapolator:
    """
    A general purpose that takes your data and just extrapolates it.
    This currently only exists in template format.
    """
    def __init__(self,
                 extrapolation_function,
                 warning=False,
                 **opts
                 ):
        """
        :param extrapolation_function: the function to handle extrapolation off the interpolation grid
        :type extrapolation_function: None | function | Callable | Interpolator
        :param warning: whether to emit a message warning about extrapolation occurring
        :type warning: bool
        :param opts: the options to feed into the extrapolator call
        :type opts:
        """
        self.extrapolator = extrapolation_function
        self.extrap_warning = warning
        self.opts = opts

    def derivative(self, n):
        return type(self)(
            self.extrapolator.derivative(n),
            warning=self.extrapolator
        )

    def find_extrapolated_points(self, gps, vals, extrap_value=np.nan):
        """
        Currently super rough heuristics to determine at which points we need to extrapolate
        :param gps:
        :type gps:
        :param vals:
        :type vals:
        :return:
        :rtype:
        """
        if extrap_value is np.nan:
            where = np.isnan(vals)
        elif extrap_value is np.inf:
            where = np.isinf(vals)
        elif not isinstance(extrap_value, (int, float, np.floating, np.integer)):
            where = np.logical_and(vals <= extrap_value[0], vals >= extrap_value[1])
        else:
            where = np.where(vals == extrap_value)

        return gps[where], where

    def apply(self, gps, vals, extrap_value=np.nan):
        ext_gps, inds = self.find_extrapolated_points(gps, vals, extrap_value=extrap_value)
        if len(ext_gps) > 0:
            new_vals = self.extrapolator(ext_gps)
            # TODO: emit a warning about extrapolating if we're doing so?
            vals[inds] = new_vals
        return vals

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)