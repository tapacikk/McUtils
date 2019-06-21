from .Graphics import Graphics, Graphics3D
import numpy as np

######################################################################################################
#
#                                    'adaptive' function sampling
#
#
def _apply_f(f, grid):
    try:
        vals = f(grid)
    except:
        vals = np.vectorize(f)(grid)

    return vals
def _semi_adaptive_sample_func(f, xmin, xmax, npts = 150, max_refines = 10, der_cut = 10^5):

    refines = 0
    der_good = False

    try:
        from ..Zachary import finite_difference
    except ImportError:
        pass
    else:
        while refines < max_refines and not der_good:
            grid = np.linspace(xmin, xmax, npts)
            vals = _apply_f(f, grid)
            ders = finite_difference(grid, vals, 1, 5)
            der_good = not np.any(abs(ders) > der_cut)
            npts *= 2
            refines += 1

    if refines == 0: # edge case
        grid = np.linspace(xmin, xmax, npts)
        vals = _apply_f(f, grid)

    return grid, vals, npts, refines
def _semi_adaptive_sample_func2(f, xmin, xmax, ymin, ymax, npts = 50, max_refines = 10, der_cut = 10^5):
    from ..Zachary import finite_difference

    refines = 0
    der_good = False
    try:
        from ..Zachary import finite_difference
    except ImportError:
        pass
    else:
        while refines < max_refines and not der_good:
            grid = np.array(np.meshgrid( np.linspace(xmin, xmax, npts), np.linspace(ymin, ymax, npts) )).T
            vals = _apply_f(f, grid)
            ders = finite_difference(grid, vals, (1, 1), (5, 5))
            der_good = not np.any(abs(ders) > der_cut)
            npts *= 2
            refines += 1

    if refines == 0: # edge case
        grid = np.linspace(xmin, xmax, npts)
        vals = _apply_f(f, grid)

    return grid, vals, npts, refines

######################################################################################################
#
#                                    Plot data methods
#
#
def _interp2DData(gpts, **opts):
    from scipy.interpolate import griddata

    x = gpts[:, 0]
    y = gpts[:, 1]

    xmin = np.min(x); xmax = np.max(x); xh = np.min(np.abs(np.diff(x)))
    ymin = np.min(y); ymax = np.max(y); yh = np.min(np.abs(np.diff(y)))

    xmesh = np.arange(xmin, xmax, xh); ymesh = np.arange(ymin, ymax, yh)
    xmesh, ymesh = np.meshgrid(xmesh, ymesh)
    mesh = np.array((xmesh, ymesh)).T
    vals = griddata(gpts[:, (0, 1)], gpts[:, 2], mesh, **opts)

    return xmesh, ymesh, vals
def _get_2D_plotdata(func, xrange):
    if not callable(func):
        fvalues = xrange
        xrange = func
    else:
        if len(xrange) == 3 and abs(xrange[2]) < abs(xrange[1] - xrange[0]):
            xrange = np.arange(*xrange)
            fvalues = _apply_f(func, xrange)
        else:
            res = _semi_adaptive_sample_func(func, *xrange)
            xrange = res[0]
            fvalues = res[1]
    return xrange, fvalues
def _get_3D_plotdata(func, xrange, yrange):
    if not callable(func):
        fvalues = yrange
        yrange = xrange
        xrange = func
    else:
        if len(xrange) == 3 and abs(xrange[2]) < abs(xrange[1] - xrange[0]):
            xrange = np.arange(*xrange)
            yrange = np.arange(*yrange)
            xrange, yrange = mesh = np.meshgrid(xrange, yrange)
            fvalues = _apply_f(func, mesh)
        else:
            res = _semi_adaptive_sample_func2(func, *xrange, *yrange)
            xrange, yrange = res[0].T
            fvalues = res[1]

    return xrange, yrange, fvalues

######################################################################################################
#
#                                    2D Plots on 2D Axes
#
#
class Plot(Graphics):
    def __init__(self, func, xrange, method = 'plot', figure = None, axes = None, **opts):
        """Creates a 1D plot on 2D axes

        :param func: either a func to apply or an array of x-values
        :type func:
        :param xrange: either an xrange spec or an array of y-values
        :type xrange:
        :param method: the method to be used to plot the data
        :type method: str
        :param opts:
        :type opts:
        """

        super().__init__(figure = figure, axes = axes)
        meth = getattr(self, method)

        xrange, fvalues = _get_2D_plotdata(func, xrange)

        meth(xrange, fvalues, **opts)
        self.set_options(**opts)
class ScatterPlot(Plot):
    def __init__(self, *args, **kwargs):
        kwargs['method'] = 'scatter'
        super().__init__(*args, **kwargs)
class HistogramPlot(Graphics):
    def __init__(self, data, **opts):
        """Creates a histogram of data

        :param data: data to histogram out
        :type data:
        :param opts:
        :type opts:
        """

        super().__init__()
        self.axes.hist(data, **opts)
        self.set_options(**opts)

######################################################################################################
#
#                                    3D Plots on 2D Axes
#
#
class Plot2D(Graphics):
    """A base class for plots that are 3D but plotted on 2D Graphics"""
    def __init__(self, func, xrange, yrange, method = 'contour', figure = None, axes = None, **opts):
        """Creates a 3D plot on 2D axes

        :param func: either a func to apply or an array of x-values
        :type func:
        :param xrange: either an xrange spec or an array of y-values
        :type xrange:
        :param yrange: either a yrange spec or an array of z-values
        :type yrange:
        :param method: the method to be used to plot the data
        :type method: str
        :param opts:
        :type opts:
        """

        super().__init__(figure = figure, axes = axes)
        meth = getattr(self, method)

        xrange, yrange, fvalues = _get_3D_plotdata(func, xrange, yrange)

        meth(xrange, yrange, fvalues, **opts)
        self.set_options(**opts)
class ContourPlot(Plot2D):
    def __init__(self, func, xrange, yrange, **opts):
        super().__init__(func, xrange, yrange, method='contour', **opts)
class DensityPlot(Plot2D):
    def __init__(self, func, xrange, yrange, **opts):
        super().__init__(func, xrange, yrange, method='pcolormesh', **opts)
class ListPlot2D(Plot2D):
    """Convenience class that handles the interpolation first"""
    def __init__(self, griddata, **opts):
        x, y, z = _interp2DData(griddata)
        super().__init__(x, y, z, **opts)
class ListContourPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='contour', **opts)
class ListDensityPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='pcolormesh', **opts)

######################################################################################################
#
#                                    3D Plots on 3D Axes
#
#
class Plot3D(Graphics3D):
    """A base class for 3D plots"""
    def __init__(self, func, xrange, yrange, method = 'plot_surface', figure = None, axes = None, **opts):
        """Creates a 3D plot on 2D axes

        :param func: either a func to apply or an array of x-values
        :type func:
        :param xrange: either an xrange spec or an array of y-values
        :type xrange:
        :param yrange: either a yrange spec or an array of z-values
        :type yrange:
        :param method: the method to be used to plot the data
        :type method: str
        :param opts:
        :type opts:
        """

        super().__init__(figure = figure, axes = axes)
        meth = getattr(self, method)

        xrange, yrange, fvalues = _get_3D_plotdata(func, xrange, yrange)

        meth(xrange, yrange, fvalues, **opts)
        self.set_options(**opts)
class ListPlot3D(Plot3D):
    """Convenience class that handles the interpolation first"""
    def __init__(self, griddata, **opts):
        x, y, z = _interp2DData(griddata)
        super().__init__(x, y, z, **opts)
class ScatterPlot3D(Plot3D):
    def __init__(self, func, xrange, yrange, **opts):
        super().__init__(func, xrange, yrange, method = 'scatter', **opts)
class WireframePlot3D(Plot3D):
    def __init__(self, func, xrange, yrange, **opts):
        super().__init__(func, xrange, yrange, method = 'plot_wireframe', **opts)
class ContourPlot3D(Plot3D):
    def __init__(self, func, xrange, yrange, **opts):
        super().__init__(func, xrange, yrange, method = 'contourf', **opts)