from .Graphics import Graphics, Graphics3D
import numpy as np
import matplotlib.figure
import matplotlib.axes

######################################################################################################
#
#                                    'adaptive' function sampling
#
#
#region function application
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
#endregion

######################################################################################################
#
#                                    Plot data methods
#
#
#region plot data prep
def _interp2DData(gpts, **opts):
    from scipy.interpolate import griddata

    x = np.sort(gpts[:, 0])
    y = np.sort(gpts[:, 1])

    xmin = np.min(x); xmax = np.max(x)
    xdiffs = np.abs(np.diff(x)); xh = np.min(xdiffs[np.nonzero(xdiffs)])
    ymin = np.min(y); ymax = np.max(y)
    ydiffs = np.abs(np.diff(y)); yh = np.min(xdiffs[np.nonzero(ydiffs)])

    num_x = (xmin - xmax) / xh
    if 5 > num_x or num_x < 10000: # don't want to get too wild
        num_x = 100 # okay but let's get a little wild
    num_y = (ymin - ymax) / yh
    if 5 > num_y or num_y < 10000: # don't want to get too wild
        num_y = 100 # okay but let's get a little wild

    # import sys
    # print(num_x, num_y, file = sys.stderr)

    xmesh = np.linspace(xmin, xmax, num_x); ymesh = np.linspace(ymin, ymax, num_y)
    xmesh, ymesh = np.meshgrid(xmesh, ymesh)
    mesh = np.array((xmesh, ymesh)).T
    vals = griddata(gpts[:, (0, 1)], gpts[:, 2], mesh, **opts)

    return xmesh, ymesh, vals.T
def _get_2D_plotdata(func, xrange):
    if not callable(func):
        fvalues = xrange
        xrange = func
    else:
        if len(xrange) == 3 and abs(xrange[2]) < abs(xrange[1] - xrange[0]):
            xrange = np.arange(*xrange)
            fvalues = _apply_f(func, xrange)
        elif len(xrange) > 2:
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
        elif len(xrange) > 2 and len(yrange) > 2:
            xrange, yrange = mesh = np.meshgrid(xrange, yrange)
            fvalues = _apply_f(func, mesh)
        else:
            res = _semi_adaptive_sample_func2(func, *xrange, *yrange)
            xrange, yrange = res[0].T
            fvalues = res[1]

    return xrange, yrange, fvalues
#endregion

######################################################################################################
#
#                                    2D Plots on 2D Axes
#
#
class Plot(Graphics):
    def __init__(self, func, xrange,
                 method = 'plot',
                 figure = None, axes = None, subplot_kw = None,
                 plot_style = None,
                 colorbar = None,
                 **opts
                 ):
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

        super().__init__(figure = figure, axes = axes, subplot_kw = subplot_kw)
        meth = getattr(self, method)

        xrange, fvalues = _get_2D_plotdata(func, xrange)

        if plot_style is None:
            plot_style = {}
        self.graphics = meth(xrange, fvalues, **plot_style)
        self.set_options(**opts)
        if colorbar:
            self.add_colorbar()
        elif isinstance(colorbar, dict):
            self.add_colorbar(**colorbar)
    def add_colorbar(self, **kw):
        fig = self.figure # type: matplotlib.figure.Figure
        ax = self.axes # type: matplotlib.axes.Axes
        fig.colorbar(self.graphics, **kw)
class ScatterPlot(Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method = "scatter", **kwargs)
class ListScatterPlot(ScatterPlot):
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)
class ErrorBarPlot(Plot):
    """A Plot object that plots error bars"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method = "errorbar", **kwargs)
class ListErrorBarPlot(ErrorBarPlot):
    """A Plot that pulls the errorbar data from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)
class StickPlot(Plot):
    """A Plot object that plots sticks"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method = "stem", **kwargs)
class ListStickPlot(StickPlot):
    """A Plot object that plots sticks from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)
class TriPlot(Plot):
    """A Plot object that plots a triangulation bars"""
    def __init__(self, *args, **opts):
        super().__init__(args, method = 'triplot', **opts)
class ListTriPlot(TriPlot):
    """A Plot that pulls the triangulation data from a list"""
    def __init__(self, griddata, **opts):
        super().__init__(griddata[:, 0], griddata[:, 1], **opts)


######################################################################################################
#
#                                    Pure Data Plots on 2D Axes
#
#
class DataPlot(Graphics):
    def __init__(self, data,
                 plot_style = None, method = None,
                 figure = None, axes = None, subplot_kw = None,
                 colorbar = None,
                 **opts
                 ):
        """Creates a plot of data
        """

        super().__init__(figure = figure, axes = axes, subplot_kw = subplot_kw)
        meth = getattr(self, method)
        if plot_style is None:
            plot_style = {}
        self.graphics = meth(data, **plot_style)
        self.set_options(**opts)
        if colorbar:
            self.add_colorbar()
        elif isinstance(colorbar, dict):
            self.add_colorbar(**colorbar)
    def add_colorbar(self, **kw):
        fig = self.figure # type: matplotlib.figure.Figure
        ax = self.axes # type: matplotlib.axes.Axes
        fig.colorbar(self.graphics, **kw)
class HistogramPlot(DataPlot):
    def __init__(*args, **kwargs):
        super().__init__(*args, method = 'hist', **kwargs)
class HistogramPlot2D(DataPlot):
    def __init__(*args, **kwargs):
        super().__init__(*args, method = 'hist2d', **kwargs)

class VerticalLinePlot(Graphics):
    def __init__(self, x, y = 1.0,
                 plot_style = None, colorbar = None,
                 figure=None, axes=None, subplot_kw = None,
                 **opts
                 ):
        """Creates a stickplot of data
        """
        if isinstance(y, (int, float)):
            y = [0, y]
        super().__init__(figure=figure, axes=axes, subplot_kw = subplot_kw)
        if plot_style is None:
            plot_style = {}
        self.graphics = self.axes.vlines(x, *y, **plot_style)
        self.set_options(**opts)
        if colorbar:
            self.add_colorbar()
        elif isinstance(colorbar, dict):
            self.add_colorbar(**colorbar)
    def add_colorbar(self, **kw):
        fig = self.figure # type: matplotlib.figure.Figure
        ax = self.axes # type: matplotlib.axes.Axes
        fig.colorbar(self.graphics, **kw)

######################################################################################################
#
#                                    3D Plots on 2D Axes
#
#
class Plot2D(Graphics):
    """A base class for plots that are 3D but plotted on 2D Graphics"""
    def __init__(self, func, xrange, yrange, plot_style = None,
                 method = 'contour', colorbar = None,
                 figure = None, axes = None, subplot_kw = None,
                 **opts
                 ):
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

        super().__init__(figure = figure, axes = axes, subplot_kw = subplot_kw)
        meth = getattr(self, method)

        xrange, yrange, fvalues = _get_3D_plotdata(func, xrange, yrange)

        if plot_style is None:
            plot_style = {}
        self.graphics = meth(xrange, yrange, fvalues, **plot_style)
        self.set_options(**opts)
        if colorbar:
            self.add_colorbar()
        elif isinstance(colorbar, dict):
            self.add_colorbar(**colorbar)

    def add_colorbar(self, **kw):
        fig = self.figure # type: matplotlib.figure.Figure
        ax = self.axes # type: matplotlib.axes.Axes
        fig.colorbar(self.graphics, **kw)
class ContourPlot(Plot2D):
    def __init__(self, func, xrange, yrange, **opts):
        super().__init__(func, xrange, yrange, method='contour', **opts)
class DensityPlot(Plot2D):
    def __init__(self, func, xrange, yrange, **opts):
        super().__init__(func, xrange, yrange, method='pcolormesh', **opts)
class ListPlot2D(Plot2D):
    """Convenience class that handles the interpolation first"""
    def __init__(self, griddata, interpolate = True, **opts):
        if interpolate:
            x, y, z = _interp2DData(griddata)
        else:
            x = griddata[:, 0]
            y = griddata[:, 1]
            z = griddata[:, 2]
        super().__init__(x, y, z, **opts)
class ListContourPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='contourf', **opts)
class ListDensityPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method='pcolormesh', **opts)
class ListTriContourPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method = 'tricontourf', interpolate = False, **opts)
class ListTriDensityPlot(ListPlot2D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method = 'tripcolor', interpolate = False, **opts)

######################################################################################################
#
#                                    3D Plots on 3D Axes
#
#
class Plot3D(Graphics3D):
    """A base class for 3D plots"""
    def __init__(self, func, xrange, yrange, plot_style = None,
                 method = 'plot_surface', colorbar = None,
                 figure = None, axes = None, subplot_kw = None,
                 **opts
                 ):
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

        super().__init__(figure = figure, axes = axes, subplot_kw = subplot_kw)
        meth = getattr(self, method)

        xrange, yrange, fvalues = _get_3D_plotdata(func, xrange, yrange)

        if plot_style is None:
            plot_style = {}
        self.graphics = meth(xrange, yrange, fvalues, **plot_style)
        self.set_options(**opts)
        if colorbar:
            self.add_colorbar()
        elif isinstance(colorbar, dict):
            self.add_colorbar(**colorbar)
    def add_colorbar(self, **kw):
        fig = self.figure # type: matplotlib.figure.Figure
        ax = self.axes # type: matplotlib.axes.Axes
        fig.colorbar(self.graphics, **kw)
class ListPlot3D(Plot3D):
    """Convenience class that handles the interpolation first"""
    def __init__(self, griddata, interpolate = True, **opts):
        if interpolate:
            x, y, z = _interp2DData(griddata)
        else:
            x = griddata[:, 0]
            y = griddata[:, 1]
            z = griddata[:, 2]
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
class ListTriPlot3D(ListPlot3D):
    def __init__(self, griddata, **opts):
        super().__init__(griddata, method = 'plot_trisurf', interpolate = False, **opts)