"""Provides a basic Plot object for working with"""

class Graphics:
    """A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with"""
    def __init__(self,
                 *args,
                 figure = None,
                 axes = None,
                 **opts
                 ):
        import matplotlib.pyplot as plt

        if figure is None:
            figure, axes = plt.subplots(*args) # yes axes is overwritten intentionally for now -- not sure how to "reparent" an Axes object
        if axes is None:
            axes = figure.add_subplot(1, 1, 1)

        self.figure = figure
        self.axes = axes

        self.set_options(**opts)

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
    @ticks.setter
    def ticks(self, ticks):
        try:
            x, y = ticks
        except ValueError:
            x, y = ticks = (self._ticks[0], ticks)

        self._ticks = ticks

        if isinstance(x, self.styled): # name feels wrong here...
            self.axes.set_xticks(*x.str, **x.opts)
        elif len(x) == 2 and not isinstance(x[0], (float, int)):
            self.axes.set_xticks(*x)
        elif x is not None:
            self.axes.set_xticks(x)
        if isinstance(y, self.styled):
            self.axes.set_yticks(*y.str, **y.opts)
        elif len(y) == 2 and not isinstance(y[0], (float, int)):
            self.axes.set_yticks(*y)
        elif y is not None:
            self.axes.set_yticks(y)

class Graphics3D:
    """A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with"""
    def __init__(self, *args, figure = None, axes = None, **opts):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        if figure is None:
            figure, axes = plt.subplots(*args, projection='3d')

        if axes is None:
            axes = axes = figure.add_subplot(1, 1, 1, projection='3d')

        self.figure = figure
        self.axes = axes

        self.set_options(**opts)

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
    @ticks.setter
    def ticks(self, ticks):
        try:
            x, y, z = ticks
        except ValueError:
            x, y, z = ticks = (self._ticks[0], self._ticks[1], ticks)

        self._ticks = ticks

        if isinstance(x, self.styled): # name feels wrong here...
            self.axes.set_xticks(*x.str, **x.opts)
        elif len(x) == 2 and not isinstance(x[0], (float, int)):
            self.axes.set_xticks(*x)
        elif x is not None:
            self.axes.set_xticks(x)
        if isinstance(y, self.styled):
            self.axes.set_yticks(*y.str, **y.opts)
        elif len(y) == 2 and not isinstance(y[0], (float, int)):
            self.axes.set_yticks(*y)
        elif y is not None:
            self.axes.set_yticks(y)
        if isinstance(z, self.styled):
            self.axes.set_Zticks(*z.str, **z.opts)
        elif len(z) == 2 and not isinstance(z[0], (float, int)):
            self.axes.set_yticks(*z)
        elif z is not None:
            self.axes.set_yticks(z)