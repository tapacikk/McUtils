"""Provides a basic Plot object for working with"""
import matplotlib.figure
import matplotlib.axes

class GraphicsException(Exception):
    pass

from abc import *
class GraphicsBase(metaclass=ABCMeta):
    def __init__(self,
                 *args,
                 figure = None,
                 axes = None,
                 subplot_kw = {},
                 **opts
                 ):
        self.figure, self.axes = self._init_suplots(figure, axes, *args, **subplot_kw)
        self._shown = False
        self.set_options(**opts)
    def _init_suplots(self, figure, axes, *args, **kw):
        """Initializes the subplots for the Graphics object

        :param figure:
        :type figure:
        :param axes:
        :type axes:
        :param args:
        :type args:
        :param kw:
        :type kw:
        :return: figure, axes
        :rtype: matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if figure is None:
            figure, axes = plt.subplots(*args, subplot_kw=kw)
            # yes axes is overwritten intentionally for now -- not sure how to "reparent" an Axes object
        elif isinstance(figure, GraphicsBase):
            axes = figure.axes # type: matplotlib.axes.Axes
            figure = figure.figure # type: matplotlib.figure.Figure

        if axes is None:
            axes = figure.add_subplot(1, 1, 1) # type: matplotlib.axes.Axes

        return figure, axes
    @abstractmethod
    def set_options(self, **opts):
        """Sets options for the plot

        :param opts:
        :type opts:
        :return:
        :rtype:
        """
        pass

    def __getattr__(self, item):
        try:
            meth = getattr(self.axes, item)
        except AttributeError as e:
            meth = getattr(self.figure, item)
        return meth

    class styled:
        def __init__(self, *str, **opts):
            self.str = str
            self.opts = opts

    def copy_axes(self):
        """Copies the axes object

        :return:
        :rtype: matplotlib.axes.Axes
        """
        raise GraphicsException("{}.{} this hack doesn't work anymore".format(
                                type(self).__name__,
                                "copy_axes"
                                ))

        import pickle, io

        buf = io.BytesIO()
        pickle.dump(self.axes, buf)
        buf.seek(0)
        return pickle.load(buf)

    def refresh(self):
        """Refreshes the axes

        :return:
        :rtype:
        """

        self.axes = self.copy_axes()
        self.figure = self.axes.figure

        return self

    def copy(self):
        """Creates a copy of the object with new axes

        :return:
        :rtype:
        """
        from copy import copy

        cp = copy(self)
        cp.axes = self.copy_axes()
        cp.figure = cp.axes

    def show(self):
        import matplotlib.pyplot as plt
        if not self._shown:
            self.figure.show()
            plt.show()
            self._shown = True
        else:
            self._shown = False
            self.refresh().show()
            #raise GraphicsException("{}.show can only be called once per object".format(type(self).__name__))

    ## useful shared bits
    def _set_ticks(self, x, set_ticks = None, set_locator = None, set_minor_locator = None, **opts):
        import matplotlib.ticker as ticks

        if isinstance(x, self.styled): # name feels wrong here...
            self._set_ticks(*x.str,
                            set_ticks = set_ticks,
                            set_locator = set_locator, set_minor_locator = set_minor_locator,
                            **x.opts
                            )
        elif isinstance(x, ticks.Locator):
            set_locator(x)
        elif isinstance(x, (list, tuple)):
            if len(x) == 2 and isinstance(x[0], (list, tuple)):
                self.axes.set_xticks(*x, **opts)
            elif len(x) == 2 and isinstance(x[0], ticks.Locator):
                set_locator(x[0])
                set_minor_locator(x[1])
        elif isinstance(x, (float, int)):
            set_ticks(ticks.MultipleLocator(x), **opts)
        elif x is not None:
            set_ticks(x, **opts)


class Graphics(GraphicsBase):
    """A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with"""

    def set_options(self,
                    axes_labels = None,
                    plot_label = None,
                    plot_range = None,
                    plot_legend = None,
                    ticks = None,
                    scale = None,
                    **ignored
                    ):

        axes = self.axes

        self._plot_label = plot_label
        if self._plot_label is not None:
            self.plot_label = plot_label
        else:
            self._plot_label = axes.get_title()

        self._plot_legend = plot_legend
        if self._plot_legend is not None:
            self.plot_legend = plot_legend
        else:
            self._plot_legend = axes.get_legend()

        self._axes_labels = axes_labels
        if self._axes_labels is not None:
            self.axes_labels = axes_labels
        else:
            self._axes_labels = (axes.get_xlabel(), axes.get_ylabel())

        self._plot_range = plot_range
        if self._plot_range is not None:
            self.plot_range = self._plot_range
        else:
            self._plot_range = (axes.get_xlim(), axes.get_ylim())

        self._ticks = ticks
        if self._ticks is not None:
            self.ticks = self._ticks
        else:
            self._ticks = (axes.get_xticks(), axes.get_yticks())

        self._scale = scale
        if self._scale is not None:
            self.scale = self._scale
        else:
            self._scale = (axes.get_xscale(), axes.get_yscale())

    # set plot label
    @property
    def plot_label(self):
        return self._plot_label
    @plot_label.setter
    def plot_label(self, label):
        self._plot_label = label
        if label is None:
            self.axes.set_title("")
        elif isinstance(label, self.styled):
            self.axes.set_title(*label.str, **label.opts)
        else:
            self.axes.set_title(label)

    # set plot legend
    @property
    def plot_legend(self):
        return self._plot_legend
    @plot_legend.setter
    def plot_legend(self, legend):
        self._plot_legend = legend
        if legend is None:
            self.axes.set_label("")
        elif isinstance(legend, self.styled):
            self.axes.set_label(*legend.str, **legend.opts)
        else:
            self.axes.set_label(legend)

    # set axes labels
    @property
    def axes_labels(self):
        return self._axes_labels
    @axes_labels.setter
    def axes_labels(self, labels):
        try:
            xlab, ylab = labels
        except ValueError:
            xlab, ylab = labels = (labels, self._axes_labels[1])

        self._axes_labels = tuple(labels)
        if xlab is None:
            self.axes.set_xlabel("")
        elif isinstance(xlab, self.styled):
            self.axes.set_xlabel(*xlab.str, **xlab.opts)
        else:
            self.axes.set_xlabel(xlab)
        if ylab is None:
            self.axes.set_ylabel("")
        elif isinstance(ylab, self.styled):
            self.axes.set_ylabel(*ylab.str, **ylab.opts)
        else:
            self.axes.set_ylabel(ylab)

    # set plot ranges
    @property
    def plot_range(self):
        return self._plot_range
    @plot_range.setter
    def plot_range(self, ranges):
        try:
            x, y = ranges
        except ValueError:
            x, y = ranges = (self._plot_range[0], ranges)
        else:
            if isinstance(x, int) or isinstance(x, float):
                x, y = ranges = (self._plot_range[0], ranges)

        self._plot_range = tuple(ranges)

        if isinstance(x, self.styled): # name feels wrong here...
            self.axes.set_xlim(*x.str, **x.opts)
        elif x is not None:
            self.axes.set_xlim(x)
        if isinstance(y, self.styled):
            self.axes.set_ylim(*y.str, **y.opts)
        elif y is not None:
            self.axes.set_ylim(y)

    # set plot ranges
    @property
    def ticks(self):
        return self._ticks
    def _set_xticks(self, x, **opts):
        return self._set_ticks(x,
                        set_ticks=self.axes.set_xticks,
                        set_locator=self.axes.xaxis.set_major_locator,
                        set_minor_locator=self.axes.xaxis.set_minor_locator,
                        **opts
                        )
    def _set_yticks(self, y, **opts):
        return self._set_ticks(y,
                               set_ticks=self.axes.set_yticks,
                               set_locator=self.axes.yaxis.set_major_locator,
                               set_minor_locator=self.axes.yaxis.set_minor_locator,
                               **opts
                               )
    @ticks.setter
    def ticks(self, ticks):

        try:
            x, y = ticks
        except ValueError:
            x, y = ticks = (self._ticks[0], ticks)

        self._ticks = ticks
        self._set_xticks(x)
        self._set_yticks(y)

class Graphics3D(GraphicsBase):
    """A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with"""

    def _init_suplots(self, figure, axes, *args, **kw):

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        if figure is None:
            figure, axes = plt.subplots(*args, subplot_kw={"projection": '3d'})
        elif isinstance(figure, GraphicsBase):
            axes = figure.axes
            figure = figure.figure

        if axes is None:
            axes = axes = figure.add_subplot(1, 1, 1, projection='3d')

        return figure, axes

    def set_options(self,
                    axes_labels = None,
                    plot_label = None,
                    plot_range = None,
                    plot_legend = None,
                    ticks = None,
                    scale = None,
                    **ignored
                    ):

        axes = self.axes

        self._plot_label = plot_label
        if self._plot_label is not None:
            self.plot_label = plot_label
        else:
            self._plot_label = self.axes.get_title()

        self._plot_legend = plot_legend
        if self._plot_legend is not None:
            self.plot_legend = plot_legend
        else:
            self._plot_legend = self.axes.get_legend()

        self._axes_labels = axes_labels
        if self._axes_labels is not None:
            self.axes_labels = axes_labels
        else:
            self._axes_labels = (axes.get_xlabel(), axes.get_ylabel(), axes.get_zlabel())

        self._plot_range = plot_range
        if self._plot_range is not None:
            self.plot_range = self._plot_range
        else:
            self._plot_range = (axes.get_xlim(), axes.get_ylim(), axes.get_zlim())

        self._ticks = ticks
        if self._ticks is not None:
            self.ticks = self._ticks
        else:
            self._ticks = (axes.get_xticks(), axes.get_yticks(), axes.get_zticks())

        self._scale = scale
        if self._scale is not None:
            self.scale = self._scale
        else:
            self._scale = (axes.get_xscale(), axes.get_yscale(), axes.get_zscale())

    # set plot label
    @property
    def plot_label(self):
        return self._plot_label
    @plot_label.setter
    def plot_label(self, label):
        self._plot_label = label
        if label is None:
            self.axes.set_title("")
        elif isinstance(label, self.styled):
            self.axes.set_title(*label.str, **label.opts)
        else:
            self.axes.set_title(label)

    # set plot legend
    @property
    def plot_legend(self):
        return self._plot_legend
    @plot_legend.setter
    def plot_legend(self, legend):
        self._plot_legend = legend
        if legend is None:
            self.axes.set_label("")
        elif isinstance(legend, self.styled):
            self.axes.set_label(*legend.str, **legend.opts)
        else:
            self.axes.set_label(legend)

    # set axes labels
    @property
    def axes_labels(self):
        return self._axes_labels
    @axes_labels.setter
    def axes_labels(self, labels):
        try:
            xlab, ylab, zlab = labels
        except ValueError:
            xlab, ylab, zlab = labels = (labels, self._axes_labels[1], self._axes_labels[2])

        self._axes_labels = tuple(labels)
        if xlab is None:
            self.axes.set_xlabel("")
        elif isinstance(xlab, self.styled):
            self.axes.set_xlabel(*xlab.str, **xlab.opts)
        else:
            self.axes.set_xlabel(xlab)
        if ylab is None:
            self.axes.set_ylabel("")
        elif isinstance(ylab, self.styled):
            self.axes.set_ylabel(*ylab.str, **ylab.opts)
        else:
            self.axes.set_ylabel(ylab)
        if zlab is None:
            self.axes.set_zlabel("")
        elif isinstance(zlab, self.styled):
            self.axes.set_ylabel(*zlab.str, **zlab.opts)
        else:
            self.axes.set_zlabel(ylab)

    # set plot ranges
    @property
    def plot_range(self):
        return self._plot_range
    @plot_range.setter
    def plot_range(self, ranges):
        try:
            x, y, z = ranges
        except ValueError:
            x, y, z = ranges = (self._plot_range[0], self._plot_range[1], ranges)
        else:
            if isinstance(x, int) or isinstance(x, float):
                x, y, z = ranges = (self._plot_range[0], self._plot_range[1], ranges)

        self._plot_range = tuple(ranges)

        if isinstance(x, self.styled): # name feels wrong here...
            self.axes.set_xlim(*x.str, **x.opts)
        elif x is not None:
            self.axes.set_xlim(x)
        if isinstance(y, self.styled):
            self.axes.set_ylim(*y.str, **y.opts)
        elif y is not None:
            self.axes.set_ylim(y)
        if isinstance(z, self.styled):
            self.axes.set_zlim(*z.str, **z.opts)
        elif z is not None:
            self.axes.set_zlim(y)

    # set plot ranges
    @property
    def ticks(self):
        return self._ticks
    def _set_xticks(self, x, **opts):
        return self._set_ticks(x,
                               set_ticks=self.axes.set_xticks,
                               set_locator=self.axes.xaxis.set_major_locator,
                               set_minor_locator=self.axes.xaxis.set_minor_locator,
                               **opts
                               )
    def _set_yticks(self, y, **opts):
        return self._set_ticks(y,
                               set_ticks=self.axes.set_yticks,
                               set_locator=self.axes.yaxis.set_major_locator,
                               set_minor_locator=self.axes.yaxis.set_minor_locator,
                               **opts
                               )
    def _set_zticks(self, z, **opts):
        return self._set_ticks(z,
                               set_ticks=self.axes.set_zticks,
                               set_locator=self.axes.zaxis.set_major_locator,
                               set_minor_locator=self.axes.zaxis.set_minor_locator,
                               **opts
                               )
    @ticks.setter
    def ticks(self, ticks):

        try:
            x, y, z = ticks
        except ValueError:
            x, y, z = ticks = (self._ticks[0], self._ticks[1], ticks)

        self._ticks = ticks

        self._set_xticks(x)
        self._set_yticks(y)
        self._set_zticks(z)