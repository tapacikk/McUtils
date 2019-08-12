"""
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class
"""

import numpy as np
import scipy.interpolate as interpolate
from .Mesh import Mesh

__all__ = [
    "Interpolator",
    "Extrapolator"
]


class InterpolatorException(Exception):
    pass


######################################################################################################
##
##                                   Interpolator Class
##
######################################################################################################
class Interpolator:
    """
    A general purpose that takes your data and just interpolates it without whining or making you do a pile of extra work
    """
    def __init__(self,
                 grid,
                 vals,
                 interpolation_function=None,
                 extrapolator=None,
                 **interpolation_opts
                 ):
        """
        :param grid: an unstructured grid of points **or** a structured grid of points **or** a 1D array
        :type grid: np.ndarray
        :param vals: the values at the grid points
        :type vals: np.ndarray
        :param interpolation_function: the basic function to be used to handle the raw interpolation
        :type interpolation_function: None | function
        :param extrapolator: the extrapolator to use for data points not on the grid
        :type extrapolator: Extrapolator
        :param interpolation_opts: the options to be fed into the interpolating_function
        :type interpolation_opts:
        """
        self.grid = grid = Mesh(grid) if not isinstance(grid, Mesh) else grid
        self.vals = vals
        self.interpolator = self.get_interpolator(grid, vals, **interpolation_opts) if interpolation_function is None else interpolation_function
        self.extrapolator = self.get_extrapolator(grid) if extrapolator is None else extrapolator

    @classmethod
    def get_interpolator(cls, grid, vals, **opts):
        """Returns a function that can be called on grid points to interpolate them

        :param grid:
        :type grid: Mesh
        :param vals:
        :type vals: np.ndarray
        :param opts:
        :type opts:
        :return: interpolator
        :rtype: function
        """
        if grid.ndim == 1:
            # 1D cases trivial with interp1D
            # should maybe handle method...?
            interpolator = interpolate.interp1d(grid, vals, **opts)
        elif grid.mesh_type == Mesh.MeshType_Structured:
            if grid.dimension == 2:
                # structured potentially 2D
                x, y = grid.gridpoints.T
                v = vals.flatten()
                # should add something for automatic method determination I think...
                interpolator = interpolate.interp2d(x, y, v, **opts)
            else:
                interpolator = interpolate.RegularGridInterpolator(grid.gridpoints, vals.flatten(), **opts)
        elif grid.mesh_type == Mesh.MeshType_Unstructured:
            # for now we'll only use the RadialBasisFunction interpolator, but this may be extended in the future
            interpolator = interpolate.Rbf(*grid.gridpoints.T, vals, **opts)
        elif grid.mesh_type == Mesh.MeshType_SemiStructured:
            # 1d Cubic extrapolator to normal grid / 1d "fill" extrapolator (uses last data point to extend to regular grid)
            # not sure what we want to do here... I'm thinking we can use some default
            # extrapolator or the Rbf to extrapolate to a full grid then from there build a RegularGridInterpolator?
            raise NotImplemented
        else:
            raise InterpolatorException("{}.{}: can't handle mesh_type '{}'".format(
                cls.__name__,
               'get_interpolator',
                grid.mesh_type
            ))

        return interpolator

    @classmethod
    def get_extrapolator(cls, grid):
        """Returns an Extrapolator that can be called on grid points to extrapolate them

        :param grid:
        :type grid: np.ndarray
        :param opts:
        :type opts:
        :return: extrapolator
        :rtype: Extrapolator
        """
        ...

    def apply(self, grid_points):
        """Interpolates then extrapolates the function at the grid_points

        :param grid_points:
        :type grid_points:
        :return:
        :rtype:
        """
        # determining what points are "inside" a region is quite tough
        # instead it is probably better to allow the basic thing to interpolate and do its thing
        # and then allow the extrapolator to post-process that result
        vals = self.interpolator(grid_points)
        return self.extrapolator(grid_points, vals)

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    # we can define a bunch of non-standard interpolators here
    @classmethod
    def morse_interpolator(cls, grid, vals, order = 4):
        """Fits data to an n-dimensional product-of-Morse oscillator potential

        :param grid:
        :type grid:
        :param vals:
        :type vals:
        :param order:
        :type order:
        :return:
        :rtype:
        """
        from scipy.optimize import curve_fit

        def morse_basis(coordinates, r0, alpha):
            ...

        def full_fun(r0s, alphas):
            ...

        data_ranges = ...
        subfits = ... # curve fit each individual morse_basis

        params = curve_fit(full_fun, data_ranges, ...)

        def morse_fit(gridpoints):
            """Applies the fitted Morse potential to the data points

            :param gridpoints:
            :type gridpoints:
            :return:
            :rtype:
            """
            ...

        return morse_fit

    @classmethod
    def regular_grid(cls, grid, vals, interp_kind='cubic', fillvalues=False, **kwargs):
        """ creates a regular grid from a set of semistructured points. Only has 2D capabilities.
        :param grid: a semistructured grid of points.
        :type grid: np.ndarray (x, y)
        :param vals: the values at the grid points.
        :type vals: np.ndarray (z)
        :param interp_kind: type of interpolation to do ('cubic' | 'linear' | 'nearest' | ...)
        :type interp_kind: str
        :param fillvalues: if true, outer edges are filled with last data point extending out.
        :type fillvalues: bool
        :param kwargs:
        :return: square_grid: a structured grid of points (np.ndarray) (x, y)
        :return: square_vals: the values at all the grid points (np.ndarray) (z)
        """
        x = grid[:, 0]
        xvals = np.unique(x)
        xyzz = np.column_stack((*grid, vals))
        slices = [xyzz[x == xv] for xv in xvals]
        maxx = max(len(slIce) for slIce in slices)
        idx = np.argmax([len(slIce) for slIce in slices])
        rnge = sorted(slices[idx][:, 1])
        sgrid = np.empty((0, 3))
        for slICE in slices:
            slICE = slICE[slICE[:, 1].argsort()]
            if len(slICE) < maxx:
                xx = np.repeat(slICE[0, 0], maxx)
                g = rnge
                if fillvalues:
                    f = interpolate.interp1d(slICE[:, 1], slICE[:, 2], kind=interp_kind,
                                             fill_value=(slICE[0, 2], slICE[-1, 2]), bounds_error=False)
                else:
                    f = interpolate.interp1d(slICE[:, 1], slICE[:, 2], kind=interp_kind,
                                             fill_value='extrapolate', bounds_error=False)
                y_fit = f(g)
                pice = np.column_stack((xx, g, y_fit))
                sgrid = np.append(sgrid, pice, axis=0)
            else:
                sgrid = np.append(sgrid, slICE, axis=0)
        square_grid = sgrid[:, :2]
        square_vals = sgrid[:, 2]
        return square_grid, square_vals
######################################################################################################
##
##                                   Extrapolator Class
##
######################################################################################################
class Extrapolator:
    """
    A general purpose that takes your data and just extrapolates it
    """
    def __init__(self,
                 extrapolation_function,
                 warning = False,
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


    def find_extrapolated_points(self, gps, vals):
        """

        :param gps:
        :type gps:
        :param vals:
        :type vals:
        :return:
        :rtype:
        """
        ...

    def apply(self, gps, vals):
        ...

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)