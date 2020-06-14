"""
Provides Graphics base classes that can be extended upon
"""
import matplotlib.figure
import matplotlib.axes

__all__ = ["GraphicsBase", "Graphics", "Graphics3D", "GraphicsGrid", "Styled"]


class GraphicsException(Exception):
    pass


########################################################################################################################
#
#                                               GraphicsBase
#
from abc import *

class GraphicsBase(metaclass=ABCMeta):
    """
    The base class for all things Graphics
    Defines the common parts of the interface with some calling into matplotlib
    """
    opt_keys = {
        'axes_labels',
        'plot_label',
        'plot_range',
        'plot_legend',
        'ticks',
        'scale',
        'image_size',
        'event_handlers',
        'animated',
        'background',
        'epilog'
        'prolog'
    }

    def __init__(self,
                 *args,
                 figure = None,
                 axes = None,
                 subplot_kw = None,
                 parent = None,
                 non_interactive = None,
                 **opts
                 ):
        """
        :param args:
        :type args:
        :param figure:
        :type figure: matplotlib.figure.Figure | None
        :param axes:
        :type axes: matplotlib.axes.Axes | None
        :param subplot_kw:
        :type subplot_kw: dict | None
        :param parent:
        :type parent: GraphicsBase | None
        :param opts:
        :type opts:
        """

        if subplot_kw is None:
            subplot_kw = {}

        self.non_interactive = non_interactive
        self.figure, self.axes = self._init_suplots(figure, axes, *args, **subplot_kw)
        self.set_options(**opts)

        self.event_handler = None
        self._shown = False
        self.parent = parent
        self.animator = None

    @staticmethod
    def _subplot_init(*args, **kw):
        import matplotlib.pyplot as plt

        return plt.subplots(*args, **kw)

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

        if figure is None:
            figure, axes = self._subplot_init(*args, **kw)
            # yes axes is overwritten intentionally for now -- not sure how to "reparent" an Axes object
        elif isinstance(figure, GraphicsBase):
            axes = figure.axes # type: matplotlib.axes.Axes
            figure = figure.figure # type: matplotlib.figure.Figure

        if axes is None:
            axes = figure.add_subplot(1, 1, 1) # type: matplotlib.axes.Axes

        return figure, axes

    @property
    def event_handlers(self):
        from .Interactive import EventHandler
        h = self.event_handler  # type: EventHandler
        if h is not None:
            h = h.data
        return h

    @property
    def animated(self):
        return self._animated

    def bind_events(self, *handlers, **events):
        from .Interactive import EventHandler

        if len(handlers) > 0 and isinstance(handlers[0], dict):
            handlers = handlers[0]
        elif len(handlers) == 0 or (len(handlers) > 0 and handlers[0] is not None):
            handlers = dict(handlers)
        if isinstance(handlers, dict):
            handlers = dict(handlers, **events)
            if self.event_handler is None:
                self.event_handler = EventHandler(self, **handlers)
            else:
                self.event_handler.bind(**handlers)

    def create_animation(self, *args, **opts):
        from .Interactive import Animator

        if len(args) > 0 and args[0] is not None:
            if self.animator is not None:
                self.animator.stop()
            self.animator = Animator(self, *args, **opts)

    def set_options(self,
                    event_handlers=None,
                    animated=None,
                    prolog=None,
                    epilog=None,
                    **opts
                    ):
        """Sets options for the plot
        :param event_handlers:
        :param animated:
        :param opts:
        :type opts:
        :return:
        :rtype:
        """
        self.bind_events(event_handlers)
        self._animated = animated
        self.create_animation(animated)

        self._prolog = prolog
        if self._prolog is not None:
            self.prolog = prolog

        self._epilog = epilog
        if self._epilog is not None:
            self.epilog = epilog

    @property
    def prolog(self):
        return self._prolog
    @prolog.setter
    def prolog(self, p):
        # might want to clear the elements in the prolog?
        self._prolog = p

    @property
    def epilog(self):
        return self._epilog
    @epilog.setter
    def epilog(self, e):
        # might want to clear the elements in the epilog?
        self._epilog = e

    def __getattr__(self, item):
        try:
            meth = getattr(self.axes, item)
        except AttributeError as e:
            meth = getattr(self.figure, item)
        return meth

    class modified:
        def __init__(self, *str, **opts):
            self.val = str
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

    @property
    def opts(self):
        # for k in self.opt_keys:
        #     getattr(self, k)
        return {k: getattr(self, k) for k in self.opt_keys if k in self.__dict__}

    def copy(self):
        """Creates a copy of the object with new axes and a new figure

        :return:
        :rtype:
        """
        return type(self)(**self.opts)

    def show(self, reshow = True):
        from .VTKInterface import VTKWindow

        if isinstance(self.figure, VTKWindow):
            self.figure.show()
        else:
            import matplotlib.pyplot as plt

            if self.non_interactive:
                plt.ioff()
            elif self.non_interactive is False:
                plt.ion()

            if reshow or not self._shown:
                self.set_options(**self.opts)  # matplotlib is dumb so it makes sense to just reset these again...
                if self.epilog is not None:
                    self._epilog_graphics = [ e.plot(self.axes) for e in self.epilog ]
                if self.prolog is not None:
                    self._prolog_graphics = [ p.plot(self.axes) for p in self.prolog ]

                plt.show()
                self._shown = True
            else:
                self._shown = False
                self.refresh().show()
                # raise GraphicsException("{}.show can only be called once per object".format(type(self).__name__))

    # useful shared bits
    def _set_ticks(self, x, set_ticks=None, set_locator=None, set_minor_locator=None, **opts):
        import matplotlib.ticker as ticks

        if isinstance(x, self.modified):  # name feels wrong here...
            self._set_ticks(*x.val,
                            set_ticks=set_ticks,
                            set_locator=set_locator, set_minor_locator=set_minor_locator,
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

    def clear(self):
        ax = self.axes  # type: matplotlib.axes.Axes
        all_things = ax.artists + ax.patches
        for a in all_things:
            a.remove()

    def _repr_html_(self):
        # hacky, but hopefully enough to make it work?
        return self.figure._repr_html_()

    def to_png(self):
        import io
        buf = io.BytesIO()
        fig = self.figure
        fig.savefig(buf,
                    format='png',
                    facecolor = self.background #-_- stupid MPL
                    )
        buf.seek(0)
        return buf

    def _repr_png_(self):
        # currently assumes a matplotlib backend...
        return self.to_png().read()

Styled = GraphicsBase.modified

########################################################################################################################
#
#                                               Graphics
#
class Graphics(GraphicsBase):
    """
    A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with
    """
    def __init__(self, *args,
                 figure=None,
                 axes=None,
                 frame=((True, False), (True, False)),
                 subplot_kw=None,
                 event_handlers=None,
                 animate=None,
                 axes_labels=None,
                 plot_label=None,
                 plot_range=None,
                 plot_legend=None,
                 ticks=None,
                 scale=None,
                 image_size=None,
                 background='white',
                 aspect_ratio = 1,
                 colorbar = None,
                 **kwargs
                 ):
        super().__init__(
            *args,
            figure=figure,
            axes=axes,
            frame=frame,
            subplot_kw=subplot_kw,
            axes_labels=axes_labels,
            plot_label=plot_label,
            plot_range=plot_range,
            plot_legend=plot_legend,
            ticks=ticks,
            scale=scale,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            event_handlers=event_handlers,
            animate=animate,
            background=background,
            colorbar = colorbar,
            **kwargs
        )

    def set_options(self,
                    axes_labels=None,
                    plot_label=None,
                    plot_range=None,
                    plot_legend=None,
                    frame=None,
                    ticks=None,
                    scale=None,
                    ticks_style=None,
                    image_size=None,
                    aspect_ratio=None,
                    background=None,
                    colorbar = None,
                    prolog = None,
                    epilog = None,
                    **parent_opts
                    ):

        super().set_options(**parent_opts)

        axes = self.axes
        self._plot_label = plot_label
        if self._plot_label is not None:
            self.plot_label = plot_label

        self._plot_legend = plot_legend
        if self._plot_legend is not None:
            self.plot_legend = plot_legend

        self._axes_labels = axes_labels
        if self._axes_labels is not None:
            self.axes_labels = axes_labels

        self._frame = frame
        if self.frame is not None:
            self.frame = frame

        self._plot_range = plot_range
        if self._plot_range is not None:
            self.plot_range = self._plot_range

        self._ticks = ticks
        if self._ticks is not None:
            self.ticks = self._ticks

        self._scale = scale
        if self._scale is not None:
            self.scale = self._scale

        self._ticks_style = ticks_style
        if ticks_style is not None:
            self.ticks_style = ticks_style

        self._image_size = image_size
        if image_size is not None:
            self.image_size = image_size

        self._aspect_ratio = aspect_ratio
        if aspect_ratio is not None:
            self.aspect_ratio = aspect_ratio

        self._background = background
        if self._background is not None:
            self.background = background

        self._colorbar = colorbar
        if self._colorbar is not None:
            self.colorbar = colorbar

    # set plot label
    @property
    def plot_label(self):
        return self._plot_label

    @plot_label.setter
    def plot_label(self, label):
        self._plot_label = label
        if label is None:
            self.axes.set_title("")
        elif isinstance(label, self.modified):
            self.axes.set_title(*label.val, **label.opts)
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
        elif isinstance(legend, self.modified):
            self.axes.set_label(*legend.val, **legend.opts)
        else:
            self.axes.set_label(legend)

    # set axes labels
    @property
    def axes_labels(self):
        return self._axes_labels

    @axes_labels.setter
    def axes_labels(self, labels):
        if self._axes_labels is None:
            self._axes_labels = (self.axes.get_xlabel(), self.axes.get_ylabel())
        try:
            xlab, ylab = labels
        except ValueError:
            xlab, ylab = labels = (labels, self._axes_labels[1])

        self._axes_labels = tuple(labels)
        if xlab is None:
            self.axes.set_xlabel("")
        elif isinstance(xlab, self.modified):
            self.axes.set_xlabel(*xlab.val, **xlab.opts)
        else:
            self.axes.set_xlabel(xlab)
        if ylab is None:
            self.axes.set_ylabel("")
        elif isinstance(ylab, self.modified):
            self.axes.set_ylabel(*ylab.val, **ylab.opts)
        else:
            self.axes.set_ylabel(ylab)

    # set plot ranges
    @property
    def plot_range(self):
        return self._plot_range

    @plot_range.setter
    def plot_range(self, ranges):
        if self._plot_range is None:
            self._plot_range = (self.axes.get_xlim(), self.axes.get_ylim())
        try:
            x, y = ranges
        except ValueError:
            x, y = ranges = (self._plot_range[0], ranges)
        else:
            if isinstance(x, int) or isinstance(x, float):
                x, y = ranges = (self._plot_range[0], ranges)

        self._plot_range = tuple(ranges)

        if isinstance(x, self.modified): # name feels wrong here...
            self.axes.set_xlim(*x.val, **x.opts)
        elif x is not None:
            self.axes.set_xlim(x)
        if isinstance(y, self.modified):
            self.axes.set_ylim(*y.val, **y.opts)
        elif y is not None:
            self.axes.set_ylim(y)

    # set plot ticks
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

        self._ticks = (self.axes.get_xticks(), self.axes.get_yticks())

        self._ticks = ticks
        self._set_xticks(x)
        self._set_yticks(y)

    # set ticks styles
    @property
    def ticks_style(self):
        return self._ticks_style

    @ticks_style.setter
    def ticks_style(self, ticks_style):
        if self._ticks_style is None:
            self._ticks_style = (None,)*2
        try:
            x, y = ticks_style
        except ValueError:
            x, y = ticks_style = (self._ticks_style[0], ticks_style)
        self._ticks_style = ticks_style
        if x is not None:
            self.axes.tick_params(
                axis='x',
                **x
            )
        if y is not None:
            self.axes.tick_params(
                axis='y',
                **y
            )

    # set size
    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, ar):
        if isinstance(ar, (float, int, str)):
            self.axes.set_aspect(ar)
        else:
            self.axes.set_aspect(ar[0], **ar[1])
    # set size
    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, wh):
        if self._image_size is None:
            self._image_size = tuple( s/72. for s in self.get_size_inches() )
        try:
            w, h = wh
        except ValueError:
            try:
                ar = self._image_size[1] / self._image_size[0]
            except TypeError:
                ar = 1
            w, h = wh = (wh, ar*wh)

        if w is not None or h is not None:
            if w is None:
                w = self._image_size[0]
            if h is None:
                h = self._image_size[1]

            if w > 72:
                wi = w/72
            else:
                wi = w
                w = 72 * w

            if h > 72:
                hi = h/72
            else:
                hi = h
                h = 72 * h

            self._image_size = (w, h)
            self.figure.set_size_inches(wi, hi)

    # set background color
    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, bg):
        self._background = bg
        self.figure.set_facecolor(bg)
        self.axes.set_facecolor(bg)

    # set show_frame
    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, fr):
        self._frame = fr
        if fr is True or fr is False:
            self.axes.set_frame_on(fr)
        else:
            lr, bt = fr
            if len(lr) == 2:
                l, r = lr
            else:
                l = lr; r=lr
            if len(bt) == 2:
                b, t = bt
            else:
                b = bt; t = bt
            self.axes.spines['left'].set_visible(l); self.axes.spines['right'].set_visible(r)
            self.axes.spines['bottom'].set_visible(b); self.axes.spines['top'].set_visible(t)

    @property
    def colorbar(self):
        return self._colorbar
    @colorbar.setter
    def colorbar(self, c):
        self._colorbar = c
        if self._colorbar is True:
            self.add_colorbar()
        elif isinstance(self._colorbar, dict):
            self.add_colorbar(**self.colorbar)
    def add_colorbar(self, graphics = None, norm = None, cmap = None, **kw):
        fig = self.figure  # type: matplotlib.figure.Figure
        ax = self.axes  # type: matplotlib.axes.Axes
        if graphics is None:
            import matplotlib.cm as cm
            graphics = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(graphics, **kw)

    # set plot scales
    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scales):
        if self._scale is None:
            self._scale = (self.axes.get_xscale(), self.axes.get_yscale())
        try:
            x, y = scales
        except ValueError:
            x, y = scales = (self._scale[0], scales)

        self._scale = tuple(scales)

        if isinstance(x, self.modified): # name feels wrong here...
            self.axes.set_xscale(*x.val, **x.opts)
        elif x is not None:
            self.axes.set_xscale(x)
        if isinstance(y, self.modified):
            self.axes.set_yscale(*y.val, **y.opts)
        elif y is not None:
            self.axes.set_yscale(y)

########################################################################################################################
#
#                                               Graphics3D
#
class Graphics3D(Graphics):
    """A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with"""
    def __init__(self, *args,
                 figure=None,
                 axes=None,
                 subplot_kw=None,
                 event_handlers=None,
                 animate=None,
                 axes_labels=None,
                 plot_label=None,
                 plot_range=None,
                 plot_legend=None,
                 ticks=None,
                 scale=None,
                 ticks_style=None,
                 image_size=None,
                 background=None,
                 backend='matplotlib',
                 **kwargs
                 ):

        self._backend = backend
        super().__init__(
            *args,
            figure=figure,
            axes=axes,
            subplot_kw=subplot_kw,
            axes_labels=axes_labels,
            plot_label=plot_label,
            plot_range=plot_range,
            plot_legend=plot_legend,
            ticks=ticks,
            scale=scale,
            ticks_style=ticks_style,
            image_size=image_size,
            event_handlers=event_handlers,
            animate=animate,
            **kwargs
        )

    @staticmethod
    def _subplot_init(*args, backend = 'MPL', **kw):
        if backend == "VTK":
            from .VTKInterface import VTKWindow
            window = VTKWindow()
            return window, window
        else:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            subplot_kw = {"projection": '3d'}
            if 'subplot_kw' in kw:
                subplot_kw = dict(subplot_kw, **kw['subplot_kw'])
                del kw['subplot_kw']
            return plt.subplots(*args, subplot_kw=subplot_kw, **kw)

    def _init_suplots(self, figure, axes, *args, **kw):
        """matplotlib subplot instantiation

        :param figure:
        :type figure:
        :param axes:
        :type axes:
        :param args:
        :type args:
        :param kw:
        :type kw:
        :return:
        :rtype:
        """

        if figure is None:
            figure, axes = self._subplot_init(*args, backend = self._backend, **kw)
        elif isinstance(figure, GraphicsBase):
            axes = figure.axes
            figure = figure.figure

        if axes is None:
            if self._backend == "VTK":
                axes = figure
            else:
                axes = figure.add_subplot(1, 1, 1, projection='3d')

        return figure, axes

    # set plot label
    @property
    def plot_label(self):
        return self._plot_label

    @plot_label.setter
    def plot_label(self, label):
        self._plot_label = label
        if label is None:
            self.axes.set_title("")
        elif isinstance(label, self.modified):
            self.axes.set_title(*label.val, **label.opts)
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
        elif isinstance(legend, self.modified):
            self.axes.set_label(*legend.val, **legend.opts)
        else:
            self.axes.set_label(legend)

    # set axes labels
    @property
    def axes_labels(self):
        return self._axes_labels

    @axes_labels.setter
    def axes_labels(self, labels):
        if self._axes_labels is None:
            self._axes_labels = (self.axes.get_xlabel(), self.axes.get_ylabel(), self.axes.get_zlabel())
        try:
            xlab, ylab, zlab = labels
        except ValueError:
            xlab, ylab, zlab = labels = (labels, self._axes_labels[1], self._axes_labels[2])

        self._axes_labels = tuple(labels)
        if xlab is None:
            self.axes.set_xlabel("")
        elif isinstance(xlab, self.modified):
            self.axes.set_xlabel(*xlab.val, **xlab.opts)
        else:
            self.axes.set_xlabel(xlab)

        if ylab is None:
            self.axes.set_ylabel("")
        elif isinstance(ylab, self.modified):
            self.axes.set_ylabel(*ylab.val, **ylab.opts)
        else:
            self.axes.set_ylabel(ylab)

        if zlab is None:
            self.axes.set_zlabel("")
        elif isinstance(zlab, self.modified):
            self.axes.set_zlabel(*zlab.val, **zlab.opts)
        else:
            self.axes.set_zlabel(zlab)

    # set plot ranges
    @property
    def plot_range(self):
        return self._plot_range

    @plot_range.setter
    def plot_range(self, ranges):

        if self._plot_range is None:
            self._plot_range = (self.axes.get_xlim(), self.axes.get_ylim(), self.axes.get_zlim())

        try:
            x, y, z = ranges
        except ValueError:
            x, y, z = ranges = (self._plot_range[0], self._plot_range[1], ranges)
        else:
            if isinstance(x, int) or isinstance(x, float):
                x, y, z = ranges = (self._plot_range[0], self._plot_range[1], ranges)


        self._plot_range = tuple(ranges)

        if isinstance(x, self.modified): # name feels wrong here...
            self.axes.set_xlim(*x.val, **x.opts)
        elif x is not None:
            self.axes.set_xlim(x)
        if isinstance(y, self.modified):
            self.axes.set_ylim(*y.val, **y.opts)
        elif y is not None:
            self.axes.set_ylim(y)
        if isinstance(z, self.modified):
            self.axes.set_zlim(*z.val, **z.opts)
        elif z is not None:
            self.axes.set_zlim(z)

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
        if self._ticks is None:
            self._ticks = (self.axes.get_xticks(), self.axes.get_yticks(), self.axes.get_zticks())
        try:
            x, y, z = ticks
        except ValueError:
            x, y, z = ticks = (self._ticks[0], self._ticks[1], ticks)

        self._ticks = ticks

        self._set_xticks(x)
        self._set_yticks(y)
        self._set_zticks(z)

    @property
    def ticks_style(self):
        return self._ticks_style

    @ticks_style.setter
    def ticks_style(self, ticks_style):
        if self._ticks_style is None:
            self._ticks_style = (None,)*3
        try:
            x, y, z = ticks_style
        except ValueError:
            x, y, z = ticks_style = (self._ticks_style[0], self._ticks_style[1], ticks_style)
        self._ticks_style = ticks_style
        if x is not None:
            self.axes.tick_params(
                axis='x',
                **x
            )
        if y is not None:
            self.axes.tick_params(
                axis='y',
                **y
            )
        if z is not None:
            self.axes.tick_params(
                axis='z',
                **z
            )

    # set size
    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, wh):
        if self._image_size is None:
            self._image_size = tuple( s/72. for s in self.get_size_inches() )
        try:
            w, h = wh
        except ValueError:
            try:
                ar = self._image_size[1] / self._image_size[0]
            except TypeError:
                ar = 1
            w, h = wh = (wh, ar*wh)

        if w is not None or h is not None:
            if w is None:
                w = self._image_size[0]
            if h is None:
                h = self._image_size[1]

            if w > 72:
                wi = w/72
            else:
                wi = w
                w = 72 * w

            if h > 72:
                hi = h/72
            else:
                hi = h
                h = 72 * h

            self._image_size = (w, h)
            self.figure.set_size_inches(wi, hi)

    # set size
    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, bg):
        self._background = bg
        self.axes.set_facecolor(bg)

    # set plot scales
    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scales):

        if self._scale is None:
            self._scale = (self.axes.get_xscale(), self.axes.get_yscale(), self.axes.get_zscale())
        try:
            x, y, z = scales
        except ValueError:
            x, y, z = scales = (self._scale[0], self._scale[1], scales)

        self._scale = tuple(scales)

        if isinstance(x, self.modified): # name feels wrong here...
            self.axes.set_xscale(*x.val, **x.opts)
        elif x is not None:
            self.axes.set_xscale(x)
        if isinstance(y, self.modified):
            self.axes.set_yscale(*y.val, **y.opts)
        elif y is not None:
            self.axes.set_yscale(y)
        if isinstance(z, self.modified):
            self.axes.set_scale(*z.val, **z.opts)
        elif y is not None:
            self.axes.set_zscale(z)


########################################################################################################################
#
#                                               GraphicsGrid
#
class GraphicsGrid:
    def __init__(self,
                 *args,
                 nrows=2, ncols=2,
                 graphics_class=Graphics,
                 figure=None,
                 axes=None,
                 subplot_kw=None,
                 _subplot_init=None,
                 tighten = False,
                 **opts
                 ):

        self.figure, self.axes = self._init_suplots(
            nrows, ncols,
            figure, axes,
            graphics_class,
            *args,
            subplot_kw=subplot_kw,
            _subplot_init=graphics_class._subplot_init if _subplot_init is None else _subplot_init,
            **opts
        )
        self.shape = (nrows, ncols)
        self._colorbar_axis = None # necessary hack for only GraphicsGrid
        self.set_options(**opts)
        if tighten:
            self.figure.tight_layout()

    def set_options(self,
                    image_size = None,
                    colorbar = None,
                    spacings = (.2, .2),
                    padding = ((.025, .2), (.125, .05)),
                    **ignored
                    ):

        self._image_size = image_size
        if image_size is not None:
            self.image_size = image_size
        else:
            first = self.axes[0]
            if not isinstance(first, GraphicsBase):
                first = first[0]
            main_size = first.figure.get_size_inches()
            width = main_size[0] * self.shape[1] * 72
            height = main_size[1] * self.shape[0] * 72
            self.image_size = (width, height)

        self._colorbar = colorbar
        if colorbar is not None:
            self.colorbar = colorbar

        self._spacings = spacings
        if spacings is not None:
            self.spacings = spacings

        self._padding = padding
        if padding is not None:
            self.padding = padding

    def _init_suplots(self,
                      nrows, ncols, figure, axes, graphics_class, *args,
                      subplot_kw=None, _subplot_init=None,
                      fig_kw=None,
                      **kw
                      ):
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

        if figure is None:
            if subplot_kw is None:
                subplot_kw = {}
            if fig_kw is None:
                fig_kw = {}
            if 'figsize' not in fig_kw:
                fig_kw['figsize'] = (4*ncols, 4*nrows)
            figure, axes = _subplot_init(*args, nrows = nrows, ncols=ncols, subplot_kw=subplot_kw, **fig_kw)

            if isinstance(axes, matplotlib.axes.Axes):
                axes = [[axes]]
            elif isinstance(axes[0], matplotlib.axes.Axes):
                if ncols == 1:
                    axes = [[ax] for ax in axes]
                else:
                    axes = [axes]

            if 'image_size' not in kw:
                kw['image_size'] = (300, 300)
            for i in range(nrows):
                for j in range(ncols):
                    axes[i][j] = graphics_class(figure=figure, axes=axes[i][j], **kw)
        elif isinstance(figure, GraphicsGrid):
            axes = figure.axes  # type: matplotlib.axes.Axes
            figure = figure.figure  # type: matplotlib.figure.Figure

        if axes is None:
            axes = [
                graphics_class(
                    figure.add_subplot(nrows, ncols, i),
                    **kw
                ) for i in range(nrows * ncols)
            ]

        return figure, axes

    def __getitem__(self, item):
        try:
            i, j = item
        except ValueError:
            return self.axes[item]
        else:
            return self.axes[i][j]

    def __setitem__(self, item, val):
        try:
            i, j = item
        except ValueError:
            self.axes[item] = val
        else:
            self.axes[i][j] = val

    # set size
    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, wh):
        try:
            w, h = wh
        except ValueError:
            try:
                ar = self._image_size[1] / self._image_size[0]
            except TypeError:
                ar = 1
            w, h = wh = (wh, ar*wh)

        if w is not None or h is not None:
            if w is None:
                w = self._image_size[0]
            if h is None:
                h = self._image_size[1]

            if w > 72:
                wi = w/72
            else:
                wi = w
                w = 72 * w

            if h > 72:
                hi = h/72
            else:
                hi = h
                h = 72 * h

            self._image_size = (w, h)
            self.figure.set_size_inches(wi, hi)

    @property
    def colorbar(self):
        return self._colorbar

    @colorbar.setter
    def colorbar(self, c):
        self._colorbar = c
        if self._colorbar is True:
            self.add_colorbar()
        elif isinstance(self._colorbar, dict):
            self.add_colorbar(**self.colorbar)

    @property
    def spacings(self):
        return self._spacings
    @spacings.setter
    def spacings(self, spacings):
        try:
            w, h = spacings
        except ValueError:
            w = h = spacings

        self._spacings = (w, h)
        self.figure.subplots_adjust(wspace=w, hspace=h)

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, padding):
        try:
            w, h = padding
        except (ValueError, TypeError):
            w = h = padding
        try:
            wx, wy = w
        except (ValueError, TypeError):
            wx = wy = w
        try:
            hx, hy = h
        except (ValueError, TypeError):
            hx = hy = h

        self._padding = ((wx, wy), (hx, hy))
        self.figure.subplots_adjust(left=wx, right=1-wy, bottom=hx, top=1-hy)

    @property
    def padding_left(self):
        return self._padding[0][0]
    @padding_left.setter
    def padding_left(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self._padding = ((p, wy), (hx, hy))
        self.figure.subplots_adjust(left=p)
    @property
    def padding_right(self):
        return self._padding[0][1]
    @padding_right.setter
    def padding_right(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self._padding = ((wx, p), (hx, hy))
        self.figure.subplots_adjust(right=1-p)
    @property
    def padding_top(self):
        return self._padding[1][1]
    @padding_top.setter
    def padding_top(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self._padding = ((wx, wy), (hx, p))
        self.figure.subplots_adjust(top=1 - p)
    @property
    def padding_bottom(self):
        return self._padding[1][0]
    @padding_bottom.setter
    def padding_bottom(self, p):
        wx, wy = self._padding[0]
        hx, hy = self._padding[1]
        self._padding = ((wx, wy), (p, hy))
        self.figure.subplots_adjust(top=p)

    def add_colorbar(self, graphics=None, norm=None, cmap=None, **kw):
        fig = self.figure  # type: matplotlib.figure.Figure
        ax = self.axes  # type: matplotlib.axes.Axes
        if graphics is None:
            import matplotlib.cm as cm
            graphics = cm.ScalarMappable(norm=norm, cmap=cmap)
        if 'cax' not in kw:
            if self._colorbar_axis is None:
                self.padding_right = .1
                self._colorbar_axis = fig.add_axes([0.93, 0.15, 0.02, 0.7])
            kw['cax'] = self._colorbar_axis

        fig.colorbar(graphics, **kw)

    def show(self):
        self.axes[0][0].show()
