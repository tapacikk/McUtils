"""
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class
"""

import numpy as np
import scipy as sp
from .Mesh import Mesh

__all__ = [
    "Interpolator",
    "Extrapolator"
]

class Interpolator:
    """
    A general purpose that takes your data and just interpolates it without whining or making you do a pile of extra work
    """
    def __init__(self,
                 grid,
                 vals,
                 interpolation_function = None,
                 extrapolator = None,
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
        self.grid = Mesh(grid) if not isinstance(grid, Mesh) else grid
        self.vals = vals
        self.interpolator = self.get_interpolator(grid, vals, **interpolation_opts) if interpolation_function is None else interpolation_function
        self.extrapolator = self.get_extrapolator(grid) if extrapolator is None else extrapolator


    @classmethod
    def get_interpolator(cls, grid, vals, **opts):
        """Returns a function that can be called on grid points to interpolate them

        :param grid:
        :type grid: np.ndarray
        :param vals:
        :type vals: np.ndarray
        :param opts:
        :type opts:
        :return: interpolator
        :rtype: function
        """
        ...

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