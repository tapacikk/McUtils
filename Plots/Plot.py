"""Provides a basic Plot object for working with"""

class Plot:
    """A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with"""
    def __init__(self,
                 *args,
                 figure = None,
                 axes = None,
                 axes_labels = None,
                 plot_label = None,
                 plot_range = None,
                 plot_legend = None,
                 ticks = None,
                 scale = None
                 ):
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        if figure is None:
            figure, axes = plt.subplots(*args) # yes axes is overwritten intentionally for now -- not sure how to "reparent" an Axes object
        if axes is None:
            axes = Axes(figure, [ [0, 1], [0, 1] ])

        self.figure = figure
        self.axes = axes

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
            xlab, ylab = labels = (labels, None)

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

        if isinstance(x, self.styled): # name feels wrong here...
            self.axes.set_xticks(*x.str, **x.opts)
        elif len(x) == 2 and not isinstance(x[0], (float, int)):
            self.axes.set_xticks(*x)
        elif x is not None:
            self.axes.set_xticks(x)
        if isinstance(y, self.styled):
            self.axes.set_yticks(*y.str, **y.opts)
        elif len(y) == 2 and not isinstance(x[y], (float, int)):
            self.axes.set_yticks(*y)
        elif y is not None:
            self.axes.set_yticks(y)

        self._ticks = (x, y)