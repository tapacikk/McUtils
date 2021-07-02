"""
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class
"""

import numpy as np, abc
import scipy.interpolate as interpolate
from .Mesh import Mesh, MeshType

__all__ = [
    "Interpolator",
    "Extrapolator",
    "RegularGridInterpolator"
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

class RegularGridInterpolator(BasicInterpolator):
    """
    A set of interpolators that support interpolation
    on a regular (tensor product) grid
    """

    def __init__(self, grids, vals, caller=None, order=3):
        """

        :param grids:
        :type grids:
        :param points:
        :type points:
        :param caller:
        :type caller:
        :param order:
        :type order:
        """
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
                caller = interpolate.InterpolatedUnivariateSpline(grids[0], vals, **opts)
            # elif ndim == 2:
            #     opts = {}
            #     if order is not None:
            #         if isinstance(order, (int, np.integer)):
            #             opts["kx"] = order
            #             opts["ky"] = order
            #         else:
            #             opts["kx"] = order[0]
            #             opts["ky"] = order[1]
            #     caller = interpolate.RectBivariateSpline(*grids, vals, **opts)
            else:
                caller = self.construct_ndspline(grids, vals, order)
        self.caller = caller

    @classmethod
    def construct_ndspline(cls, grids, vals, order):
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
            # print("!!!!!", o)
            for e,v in enumerate(coeffs):
                ppoly = interpolate.PPoly.from_spline(interpolate.splrep(g, v, k=o))
                x[i] = ppoly.x
                sub_coeffs[e] = ppoly.c
                # we now have the interpolation coefficients for a polynomial in
                # each of the flattened coeff coordinates
            coeffs = np.array(sub_coeffs)
            # print(">>", coeffs.shape, len(g))
            coeffs = coeffs.reshape(
                og_shape[1:]+
                    sub_coeffs[0].shape
            )
            # print(coeffs.shape, ndim-1, ">", ndim-1-i)
            tot_dim = ndim+i+1
            coeffs = np.moveaxis(coeffs, tot_dim-2, tot_dim-2-i)
            # we turn one index into two but need to move the axes up
            # based on the shape needed by NdPPoly
            # coeffs[]
            # print("<", coeffs.shape)

        return interpolate.NdPPoly(coeffs, x)

    def __call__(self, *args, **kwargs):
        return self.caller(*args, **kwargs)

    def derivative(self, order):
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

class Interpolator:
    """
    A general purpose that takes your data and just interpolates it without whining or making you do a pile of extra work
    """
    def __init__(self,
                 grid,
                 vals,
                 interpolation_function=None,
                 interpolation_order=None,
                 extrapolator=None,
                 extrapolation_order=1,
                 **interpolation_opts
                 ):
        """
        :param grid: an unstructured grid of points **or** a structured grid of points **or** a 1D array
        :type grid: np.ndarray
        :param vals: the values at the grid points
        :type vals: np.ndarray
        :param interpolation_function: the basic function to be used to handle the raw interpolation
        :type interpolation_function: None | function
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
            interpolation_function = self.get_interpolator(grid, vals, interpolation_order=interpolation_order, **interpolation_opts)
        self.interpolator = interpolation_function
        if extrapolator is not None:
            if extrapolator == 'Automatic':
                extrapolator = self.get_extrapolator(grid, vals, extrapolation_order=extrapolation_order)
            elif not isinstance(extrapolator, Extrapolator):
                extrapolator = Extrapolator(extrapolator)
        self.extrapolator = extrapolator

    @classmethod
    def get_interpolator(cls, grid, vals, interpolation_order = None, **opts):
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
            # 1D cases trivial with interp1D
            # should maybe handle method...?
            if interpolation_order is not None:
                opts['kind'] = interpolation_order
            if 'bounds_error' not in opts:
                opts['bounds_error'] = False
            interpolator = interpolate.UnivariateSpline(grid, vals, **opts)
        elif grid.mesh_type == MeshType.Structured:
            if grid.dimension == 2:
                # structured potentially 2D
                x, y = grid.gridpoints.T
                v = vals.flatten()
                # should add something for automatic method determination I think...
                if interpolation_order is not None:
                    opts['kind'] = interpolation_order
                interpolator = interpolate.interp2d(x, y, v, **opts)
            else:
                if interpolation_order is not None:
                    if isinstance(interpolation_order, int):
                        if interpolation_order == 1:
                            interpolation_order = "linear"
                        else:
                            raise InterpolatorException("Interpolator '{}' doesn't support interpolation order '{}'".format(
                                interpolate.RegularGridInterpolator,
                                interpolation_order
                            ))
                    opts['kind'] = interpolation_order
                interpolator = interpolate.RegularGridInterpolator(grid.gridpoints, vals.flatten(), **opts)
        elif grid.mesh_type == MeshType.Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            if interpolation_order is not None:
                if isinstance(interpolation_order, int):
                    if interpolation_order == 1:
                        interpolation_order = "linear"
                    elif interpolation_order == 3:
                        interpolation_order = "cubic"
                    elif interpolation_order == 5:
                        interpolation_order = "quintic"
                    else:
                        raise InterpolatorException("Interpolator '{}' doesn't support interpolation order '{}'".format(
                            interpolate.Rbf,
                            interpolation_order
                        ))
                opts['method'] = interpolation_order
            interpolator = interpolate.Rbf(*grid.gridpoints.T, vals, **opts)
        elif grid.mesh_type == MeshType.SemiStructured:
            if interpolation_order is not None:
                if isinstance(interpolation_order, int):
                    if interpolation_order == 1:
                        interpolation_order = "linear"
                    elif interpolation_order == 3:
                        interpolation_order = "cubic"
                    else:
                        raise InterpolatorException("Interpolator '{}' doesn't support interpolation order '{}'".format(
                            interpolate.griddata,
                            interpolation_order
                        ))
                opts['method'] = interpolation_order
            def interpolator(xi, g=grid, v=vals, opts=opts):
                return interpolate.griddata(v, g, xi, **opts)
            # # 1d Cubic extrapolator to normal grid / 1d "fill" extrapolator (uses last data point to extend to regular grid)
            # # not sure what we want to do here... I'm thinking we can use some default
            # # extrapolator or the Rbf to extrapolate to a full grid then from there build a RegularGridInterpolator?
            # raise NotImplemented
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return interpolator

    @classmethod
    def get_extrapolator(cls, grid, vals, extrapolation_order=2):
        """
        Returns an Extrapolator that can be called on grid points to extrapolate them

        :param grid:
        :type grid: np.ndarray
        :param extrapolation_order:
        :type extrapolation_order: int
        :return: extrapolator
        :rtype: Extrapolator
        """
        #TODO: turns out this works for Mathematica...but scipy isn't half so sophisticated

        return Extrapolator(
            cls(
                grid,
                vals,
                interpolation_order=1,
                extrapolator=Extrapolator(lambda g:np.full(g.shape, np.nan))
            )
        )

    def apply(self, grid_points, **opts):
        """Interpolates then extrapolates the function at the grid_points

        :param grid_points:
        :type grid_points:
        :return:
        :rtype:
        """
        # determining what points are "inside" a region is quite tough
        # instead it is probably better to allow the basic thing to interpolate and do its thing
        # and then allow the extrapolator to post-process that result
        vals = self.interpolator(grid_points, **opts)
        if self.extrapolator is not None:
            print(self.extrapolator)
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

        if self.grid.ndim == 1:
            return self.interpolator.derivative(order)
        else:
            ...


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
    As I do more work with the Surface stuff I'm sure this will get filled out more.
    One big target is to use
    """
    def __init__(self,
                 extrapolation_function,
                 warning=False,
                 **opts
                 ):
        """

        :param extrapolation_function: the function to handle extrapolation off the interpolation grid
        :type extrapolation_function: None | function
        :param warning: whether to emit a message warning about extrapolation occurring
        :type warning: bool
        :param opts: the options to feed into the extrapolator call
        :type opts:
        """
        self.extrapolator = extrapolation_function
        self.extrap_warning = warning
        self.opts = opts

    def find_extrapolated_points(self, gps, vals, extrap_value = np.nan):
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

    def extrap2d(self, gps, vals, extrap_kind='linear'):
        """ Takes a regular grid and creates a function for interpolation/extrapolation.
        :param gps: x, y data
        :type gps: ndarray
        :param vals: z data
        :type vals: ndarray
        :param extrap_kind: type of interpolation to do ('cubic' | 'linear' | 'nearest' | ...)
        :type extrap_kind: str
        :param fillvalues: if true, outer edges are filled with last data point extending out.
         Otherwise extrapolates according to extrap_kind (default)
        :type fillvalues: bool
        :return: pf: function fit to grid points for evaluation.
        :rtype: function
        """
        xx = np.unique(gps[:, 0])
        yy = np.unique(gps[:, 1])
        extrap_func = interpolate.interp2d(xx, yy, vals, kind=extrap_kind, fill_value=None)

        def pf(grid=None, x=None, y=None, extrap=extrap_func):
            if grid is not None:
                x = np.unique(grid[:, 0])
                y = np.unique(grid[:, 1])
            pvs = extrap(x, y).T
            return pvs.flatten()

        return pf

    def apply(self, gps, vals, extrap_value = np.nan):
        ext_gps, inds = self.find_extrapolated_points(gps, vals, extrap_value=extrap_value)
        if len(ext_gps) > 0:
            new_vals = self.extrapolator(ext_gps)
            # TODO: emit a warning about extrapolating if we're doing so
            vals[inds] = new_vals
        return vals

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)