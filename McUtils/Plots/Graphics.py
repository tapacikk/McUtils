"""
Provides Graphics base classes that can be extended upon
"""


import platform, os
# import matplotlib.figure
# import matplotlib.axes
import weakref

from .Properties import GraphicsPropertyManager, GraphicsPropertyManager3D
from .Styling import Styled, ThemeManager
from .Backends import Backends

__all__ = ["GraphicsBase", "Graphics", "Graphics3D", "GraphicsGrid"]


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
        'background',
        'axes_labels',
        'plot_label',
        'plot_range',
        'plot_legend',
        'legend_style',
        'ticks',
        'ticks_style',
        'ticks_label_style',
        'scale',
        'padding',
        'spacings',
        'aspect_ratio',
        'image_size',
        'event_handlers',
        'animated',
        'epilog'
        'prolog',
        'subplot_kw',
        'theme'
    }
    def _get_def_opt(self, key, val, theme):
        if val is None:
            try:
                v = object.__getattribute__(self, '_'+key) # we overloaded getattr
            except AttributeError as e:
                try:
                    v = object.__getattribute__(self._prop_manager, '_' + key)  # we overloaded getattr
                except AttributeError:
                    v = None
            if v is None and key in self.default_style:
                v = self.default_style[key]
            if v is None and theme is not None and key in theme:
                v = theme[key]
            return v
        else:
            return val

    def _update_copy_opt(self, key, val):
        if not self._in_init:
            if key not in self._init_opts or self._init_opts[key] is None:
                if val != self._get_def_opt(key, None, self.theme):
                    self._init_opts[key] = val
            elif val != self._init_opts[key]:
                self._init_opts[key] = val
        # if key in self._init_opts:
        #     print(key, self._init_opts[key])

    layout_keys = {
        'figure',
        'tighten',
        'axes',
        'subplot_kw',
        'parent',
        'image_size',
        'padding',
        'aspect_ratio',
        'interactive',
        'mpl_backend',
        'theme',
        'prop_manager',
        'theme_manager',
        'managed',
        'event_handlers',
        'animated',
        'prolog',
        'epilog'
    }
    def __init__(self,
                 *args,
                 figure=None,
                 tighten=False,
                 axes=None,
                 subplot_kw=None,
                 parent=None,
                 image_size=None,
                 padding=None,
                 aspect_ratio=None,
                 interactive=None,
                 mpl_backend=None,
                 theme=None,
                 prop_manager=GraphicsPropertyManager,
                 theme_manager=ThemeManager,
                 managed=None,
                 strict=True,
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


        self._init_opts = dict(opts, # for copying
                               tighten=tighten,
                               axes=axes,
                               subplot_kw=subplot_kw,
                               image_size=image_size,
                               padding=padding,
                               aspect_ratio=aspect_ratio,
                               interactive=interactive,
                               mpl_backend=mpl_backend,
                               theme=theme,
                               prop_manager=prop_manager,
                               theme_manager=theme_manager,
                               managed=managed
                               )
        self._in_init = True

        # we try to only load pyplot once and use it
        # as an API rather than a global import
        # this is to potentially support non-MPL interfaces
        # in the future
        # self.pyplot = None
        self._mpl_loaded = False

        if subplot_kw is None:
            subplot_kw = {}

        self.interactive = interactive
        self.mpl_backend = mpl_backend
        if isinstance(figure, GraphicsBase):
            parent = figure
        self.parent = parent
        self.children = weakref.WeakSet()
        if parent is not None:
            parent.children.add(self)
            # TODO: generalize this to a set of inherited props...
            if image_size is None: # gotta copy in some layout stuff..
                image_size = self.parent.image_size
            if aspect_ratio is None:
                aspect_ratio = self.parent.aspect_ratio
            if interactive is None:
                interactive = self.parent.interactive
            prop_manager = self.parent._prop_manager

        theme = self._get_def_opt('theme', theme, {})
        self.theme=theme
        self.theme_manager=theme_manager
        # if theme is not None:
        #     theme_dict = theme_manager.resolve_theme(theme)[1]
        # else:
        #     theme_dict = {}

        if managed is None:
            if figure is not None and isinstance(figure, GraphicsBase):
                managed = figure.managed
            else:
                managed = False
        self.managed = managed

        interactive = self._get_def_opt('interactive', interactive, theme)
        self.interactive = interactive

        aspect_ratio = self._get_def_opt('aspect_ratio', aspect_ratio, theme)
        image_size = self._get_def_opt('image_size', image_size, theme)
        padding = self._get_def_opt('padding', padding, theme)
        if figure is None and image_size is not None and 'figsize' not in subplot_kw:
            try:
                w, h = image_size
            except TypeError:
                w = image_size
                asp = aspect_ratio
                if asp is None or isinstance(asp, str):
                    asp = 4.8/6.4
                h = w * asp
            if padding is not None:
                pw, ph = padding
                try:
                    pwx, pwy = pw
                except (TypeError, ValueError):
                    pwx = pwy = pw
                try:
                    phx, phy = ph
                except (TypeError, ValueError):
                    phx = phy = ph
                w += pwx + pwy
                h += phx + phy
                image_size = (w, h)
            subplot_kw['figsize'] = (w/72., h/72.)

        self.subplot_kw = subplot_kw
        if theme is not None:
            with theme_manager.from_spec(theme):
                self.figure, self.axes = self._init_suplots(figure, axes, *args, **subplot_kw)
        else:
            self.figure, self.axes = self._init_suplots(figure, axes, *args, **subplot_kw)
        if not self.interactive:
            self.pyplot.mpl_disconnect()

        if isinstance(prop_manager, type):
            prop_manager = prop_manager(self, self.figure, self.axes, managed=managed)
        self._prop_manager = prop_manager
        self._colorbar_axis = None
        self.set_options(padding=padding, aspect_ratio=aspect_ratio, image_size=image_size, strict=strict, **opts)

        self.event_handler = None
        self._shown = False
        self.animator = None
        self.tighten = tighten


        self._in_init = False

    class MPLManager:
        _plt = None
        _mpl = None
        def __init__(self, figure):
            self.fig = figure
            self.fig.mpl = self
            self.jlab = None
            self._old_interactive = None
            self._drawer = None
            self._draw_all = None
            self._old_show = None
            self._old_mpl_backend = None
            self._old_magic_backend = None
        @property
        def plt(self):
            if self._plt is None:
                import matplotlib.pyplot as plt
                type(self)._plt = plt
            return self._plt
        @property
        def mpl(self):
            if self._mpl is None:
                import matplotlib as mpl
                type(self)._mpl = mpl
            return self._mpl
        def draw_if_interactive(self, *args, **kwargs):
            pass
        def magic_backend(self, backend):
            try:
                from IPython.core.getipython import get_ipython
            except ImportError:
                pass
            else:
                shell = get_ipython()
                ip_name = type(shell).__name__
                in_nb = ip_name == 'ZMQInteractiveShell'
                if in_nb:
                    try:
                        from IPython.core.magics import PylabMagics
                    except ImportError:
                        pass
                    else:
                        set_jupyter_backend = PylabMagics(shell).matplotlib
                        set_jupyter_backend(backend)

        def __enter__(self):
            if self.fig.mpl_backend is not None:
                self._old_mpl_backend = self.mpl.get_backend()
                self.mpl.use(self.fig.mpl_backend)
            self._was_interactive = self.plt.isinteractive()
            if not self.fig.managed:
                # import matplotlib.pyplot as plt
                # plt.ioff
                # if 'inline' in self.mpl.get_backend():
                #     backend = self.plt._backend_mod
                #     self.plt.show = ...
                #     self._old_show = backend.show
                #     backend.show = self.jupyter_show
                if not self.fig.interactive:
                    self.plt.ioff()
                    # manager.canvas.mpl_disconnect(manager._cidgcf)
                    # self._drawer = self.plt.draw_if_interactive
                    # self._draw_all = self.plt.draw_all
                    # self.plt.draw_if_interactive = self.draw_if_interactive
                    # self.plt.draw_all = self.draw_if_interactive
                    # if self.fig.mpl_backend is None:
                    #     self._old_magic_backend = self.mpl.get_backend()
                    #     self.magic_backend('Agg')
                else:
                    self.plt.ion()
                    # if self.fig.mpl_backend is None:
                    #     self._old_magic_backend = self.mpl.get_backend()
                    #     if 'inline' not in self._old_magic_backend:
                    #         self.magic_backend('inline')
            self.fig._mpl_loaded=True
            return self.plt
        def mpl_disconnect(self):
            # this is a hack that might need to be updated in the future
            if 'inline' in self.mpl.get_backend():
                try:
                    from matplotlib._pylab_helpers import Gcf
                    canvas = self.fig.figure.canvas
                    num = canvas.manager.num
                    if all(hasattr(num, attr) for attr in ["num", "_cidgcf", "destroy"]):
                        manager = num
                        if Gcf.figs.get(manager.num) is manager:
                            Gcf.figs.pop(manager.num)
                        else:
                            return
                    else:
                        try:
                            manager = Gcf.figs.pop(num)
                        except KeyError:
                            return
                    # manager.canvas.mpl_disconnect(manager._cidgcf)
                    # self.fig.figure.canvas.mpl_disconnect(
                    #     self.fig.figure.canvas.manager._cidgcf
                    # )
                except:
                    pass

        def mpl_connect(self):
            if 'inline' in self.mpl.get_backend():
                # try:
                from matplotlib._pylab_helpers import Gcf
                canvas = self.fig.figure.canvas
                manager = canvas.manager
                num = canvas.manager.num
                Gcf.figs[num] = manager
                manager._cidgcf = canvas.mpl_connect(
                    "button_press_event", lambda event: Gcf.set_active(manager)
                )
                # manager.canvas.mpl_disconnect(manager._cidgcf)
                # self.fig.figure.canvas.mpl_disconnect(
                #     self.fig.figure.canvas.manager._cidgcf
                # )
                # except:
                #     pass

        def jupyter_show(self, close=None, block=None):
            """Show all figures as SVG/PNG payloads sent to the IPython clients.
            Parameters
            ----------
            close : bool, optional
                If true, a ``plt.close('all')`` call is automatically issued after
                sending all the figures. If this is set, the figures will entirely
                removed from the internal list of figures.
            block : Not used.
                The `block` parameter is a Matplotlib experimental parameter.
                We accept it in the function signature for compatibility with other
                backends.
            """
            from matplotlib._pylab_helpers import Gcf
            from IPython.core.display import display
            mpl_inline = self.plt._backend_mod
            # from matplotlib.backends import backend as mpl_inline
            print(mpl_inline)
            # mpl_inline = sys.modules[type(self.plt).__module__]

            if close is None:
                close = mpl_inline.InlineBackend.instance().close_figures
            try:
                for figure_manager in [Gcf.get_active()]:
                    display(
                        figure_manager.canvas.figure,
                        metadata=mpl_inline._fetch_figure_metadata(figure_manager.canvas.figure)
                    )
            finally:
                self._to_draw = []
                # only call close('all') if any to close
                # close triggers gc.collect, which can be slow
                if close and Gcf.get_all_fig_managers():
                    self.plt.close('all')
        # This flag will be reset by draw_if_interactive when called
        _draw_called = False
        # list of figures to draw when flush_figures is called
        _to_draw = []
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._old_mpl_backend is not None:
                self.mpl.use(self._old_mpl_backend)
            if self._drawer is not None:
                self.plt.draw_if_interactive = self._drawer
                self._drawer = None
            if self._draw_all is not None:
                self.plt.draw_all = self._drawer
                self._drawer = None
            if self._old_show is not None:
                self.plt._backend_mod.show = self._old_show
                self._old_show = None

            if self._old_magic_backend is not None:
                if 'inline' in self._old_magic_backend:
                    self.magic_backend('inline')
                else:
                    self.mpl.use(self._old_mpl_backend)
            if self._was_interactive and not self.plt.isinteractive():
                self.plt.ion()
    @property
    def pyplot(self):
        return self.MPLManager(self)

    @staticmethod
    def _subplot_init(*args, mpl_backend=None, figure=None, **kw):
        with figure.pyplot as plt:
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
            figure, axes = self._subplot_init(*args, mpl_backend=self.mpl_backend, figure=self, **kw)
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

    def _check_opts(self, opts):
        diff = opts.keys() - self.layout_keys
        if len(diff) > 0:
            raise ValueError("unknown options for {}: {}".format(
                type(self).__name__, list(diff)
            ))
    def set_options(self,
                    event_handlers=None,
                    animated=None,
                    prolog=None,
                    epilog=None,
                    strict=True,
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
        if strict:
            self._check_opts(opts)

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

    # def __getattr__(self, item):
    #     reraise_error = None
    #     axes = object.__getattribute__(self, 'axes')
    #     try:
    #         meth = getattr(axes, item)
    #     except AttributeError as e:
    #         if 'Axes' not in e.args[0] or item not in e.args[0]:
    #             reraise_error = e
    #         else:
    #             try:
    #                 meth = getattr(self.figure, item)
    #             except AttributeError as e:
    #                 if 'Figure' not in e.args[0] or item not in e.args[0]:
    #                     reraise_error = e
    #                 else:
    #                     reraise_error = AttributeError("'{}' object has no attribute '{}'".format(
    #                         type(self).__name__,
    #                         item
    #                     ))
    #     if reraise_error is not None:
    #         raise reraise_error
    #
    #     return meth

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
        opt_dict = {}
        for k in self.opt_keys:
            if k in self.__dict__ or hasattr(self._prop_manager, k):
                opt_dict[k] = getattr(self, k)
            elif "_"+k in self.__dict__:
                opt_dict[k] = getattr(self, "_" + k)
        return opt_dict

    def copy(self):
        """Creates a copy of the object with new axes and a new figure

        :return:
        :rtype:
        """
        return self.change_figure(None)
    def change_figure(self, new, *init_args, **init_kwargs):
        """Creates a copy of the object with new axes and a new figure

        :return:
        :rtype:
        """
        # print(self._init_opts)
        return type(self)(*init_args, **dict(self._init_opts, figure=new, **init_kwargs))

    def prep_show(self):
        self.set_options(**self.opts)  # matplotlib is dumb so it makes sense to just reset these again...
        if self.epilog is not None:
            self._epilog_graphics = [e.plot(self.axes) for e in self.epilog]
        if self.prolog is not None:
            self._prolog_graphics = [p.plot(self.axes) for p in self.prolog]
        if self.tighten:
            self.figure.tight_layout()
        self._shown = True
    def show(self, reshow=False):
        from .VTKInterface import VTKWindow

        if isinstance(self.figure, VTKWindow):
            self.figure.show()
        else:
            if reshow or not self._shown:
                self.prep_show()
                if not self.managed:
                    ni = not self.interactive
                    try:
                        with self.pyplot as plt:
                            if ni:
                                self.pyplot.mpl_connect()
                            plt.show()
                    finally:
                        self.interactive = not ni
                        if ni:
                            self.pyplot.mpl_disconnect()
            else:
                # self._shown = False
                self.copy().show()
                # raise GraphicsException("{}.show can only be called once per object".format(type(self).__name__))

    def close(self, force=False):
        if force or self.parent is None: # parent manages cleanup
            if self._mpl_loaded:
                with self.pyplot as plt:
                    return plt.close(self.figure)

    def __del__(self):
        self.close()

    def clear(self):
        ax = self.axes  # type: matplotlib.axes.Axes
        all_things = ax.artists + ax.patches
        for a in all_things:
            a.remove()

    def _ipython_display_(self):
        self.show()
    def _repr_html_(self):
        # hacky, but hopefully enough to make it work?
        return self.figure._repr_html_()

    def savefig(self, where, format=None, **kw):
        """
        Saves the image to file
        :param where:
        :type where:
        :param format:
        :type format:
        :param kw:
        :type kw:
        :return: file it was saved to (I think...?)
        :rtype: str
        """
        if 'facecolor' not in kw:
            kw['facecolor'] = self.background# -_- stupid MPL
        if format is None:
            format = os.path.splitext(where)[1].split('.')[-1]
        return self.figure.savefig(where,
                    format=format,
                    **kw
                    )
    def to_png(self):
        """
        Used by Jupyter and friends to make a version of the image that they can display, hence the extra 'tight_layout' call
        :return:
        :rtype:
        """
        import io
        buf = io.BytesIO()
        self.prep_show()
        self.figure.tight_layout()
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

    def add_colorbar(self,
                     graphics=None,
                     norm=None,
                     cmap=None,
                     size=(20, 200),
                     tick_padding=40,
                     **kw
                     ):
        fig = self.figure  # type: matplotlib.figure.Figure
        ax = self.axes  # type: matplotlib.axes.Axes
        if graphics is None:
            import matplotlib.cm as cm
            graphics = cm.ScalarMappable(norm=norm, cmap=cmap)

        if 'cax' not in kw:
            if self._colorbar_axis is None:
                # TODO: I'd like to have better control over how much space this colorbar takes in the future
                #   it might be a mistake to make this not percentage based...
                W, H = self.image_size
                if size[0] < 1:
                    size = (W * size[0], H * size[1])
                size = (size[0] + tick_padding, size[1])
                cur_padding = self.padding
                if cur_padding[0][1] < size[0] + 5:
                    self.padding_right = size[0] + 5
                    new_padding = self.padding
                    wpad_old = cur_padding[0][0] + cur_padding[0][1]
                    wpad_new = new_padding[0][0] + new_padding[0][1]
                    wdiff = wpad_new - wpad_old
                else:
                    wdiff = 0
                # we need to now shrink the spacing by enough to compensate for this
                # it would be best to mess with the figure sizes themselves, but this is the easier
                # solution for the moment...
                sp = self.spacings
                if sp is not None:
                    ws, hs = sp
                    nspaces = self.shape[1] - 1
                    ws = ws - (wdiff / nspaces)
                    ws = max(ws, 0)
                    self.spacings = (ws, hs)
                cbw = size[0] / W
                tw = tick_padding / W
                cbh = size[1] / H
                xpos = 1 - cbw
                ypos = .5 - cbh / 2  # new_padding[1][0]/H

                theme = self.theme
                if self.theme is not None:
                    with self.theme_manager.from_spec(theme):
                        self._colorbar_axis = fig.add_axes([xpos, ypos, cbw - tw, cbh])
                else:
                    self._colorbar_axis = fig.add_axes([xpos, ypos, cbw - tw, cbh])
            kw['cax'] = self._colorbar_axis

        return fig.colorbar(graphics, **kw)

########################################################################################################################
#
#                                               Graphics
#
class Graphics(GraphicsBase):
    """
    A mini wrapper to matplotlib.pyplot to create a unified interface I know how to work with
    """
    default_style = dict(
        theme='mccoy',
        frame=((True, False), (True, False)),
        aspect_ratio=1,
        image_size=300,
        padding=((60, 10), (35, 10)),
        interactive=False
    )

    layout_keys = {
        'plot_label',
        'plot_legend',
        'legend_style'
        'axes_labels',
        'frame',
        'frame_style',
        'plot_range',
        'ticks',
        'ticks_style',
        'ticks_label_style',
        'scale',
        'aspect_ratio'
        'image_size',
        'padding',
        'spacings',
        'background',
        'colorbar',
    } | GraphicsBase.layout_keys
    def set_options(self,
                    axes_labels=None,
                    plot_label=None,
                    plot_range=None,
                    plot_legend=None,
                    legend_style=None,
                    frame=None,
                    frame_style=None,
                    ticks=None,
                    scale=None,
                    padding=None,
                    spacings=None,
                    ticks_style=None,
                    ticks_label_style=None,
                    image_size=None,
                    aspect_ratio=None,
                    background=None,
                    colorbar=None,
                    prolog=None,
                    epilog=None,
                    **parent_opts
                    ):

        super().set_options(**parent_opts)

        opts = (
            ('plot_label', plot_label),
            ('plot_legend', plot_legend),
            ('legend_style', legend_style),
            ('axes_labels', axes_labels),
            ('frame', frame),
            ('frame_style', frame_style),
            ('plot_range', plot_range),
            ('ticks', ticks),
            ('ticks_style', ticks_style),
            ('ticks_label_style', ticks_label_style),
            ('scale', scale),
            ('aspect_ratio', aspect_ratio),
            ('image_size', image_size),
            ('padding', padding),
            ('spacings', spacings),
            ('background', background),
            ('colorbar', colorbar)
        )
        for oname, oval in opts:
            oval = self._get_def_opt(oname, oval, {})
            if oval is not None:
                setattr(self, oname, oval)

    @property
    def artists(self):
        return []

    # attaching custom property setters
    @property
    def plot_label(self):
        return self._prop_manager.plot_label
    @plot_label.setter
    def plot_label(self, value):
        self._update_copy_opt('plot_label', value)
        self._prop_manager.plot_label = value

    @property
    def plot_legend(self):
        return self._prop_manager.plot_legend
    @plot_legend.setter
    def plot_legend(self, value):
        self._update_copy_opt('plot_legend', value)
        self._prop_manager.plot_legend = value

    @property
    def axes_labels(self):
        return self._prop_manager.axes_labels
    @axes_labels.setter
    def axes_labels(self, value):
        self._update_copy_opt('axes_labels', value)
        self._prop_manager.axes_labels = value

    @property
    def frame(self):
        return self._prop_manager.frame
    @frame.setter
    def frame(self, value):
        self._update_copy_opt('frame', value)
        self._prop_manager.frame = value

    @property
    def frame_style(self):
        return self._prop_manager.frame_style
    @frame_style.setter
    def frame_style(self, value):
        self._update_copy_opt('frame_style', value)
        self._prop_manager.frame_style = value

    @property
    def plot_range(self):
        return self._prop_manager.plot_range
    @plot_range.setter
    def plot_range(self, value):
        self._update_copy_opt('plot_range', value)
        self._prop_manager.plot_range = value

    @property
    def ticks(self):
        return self._prop_manager.ticks
    @ticks.setter
    def ticks(self, value):
        self._update_copy_opt('ticks', value)
        self._prop_manager.ticks = value

    @property
    def ticks_style(self):
        return self._prop_manager.ticks_style
    @ticks_style.setter
    def ticks_style(self, value):
        self._update_copy_opt('ticks_style', value)
        self._prop_manager.ticks_style = value

    @property
    def ticks_label_style(self):
        return self._prop_manager.ticks_label_style
    @ticks_label_style.setter
    def ticks_label_style(self, value):
        self._update_copy_opt('ticks_label_style', value)
        self._prop_manager.ticks_label_style = value

    @property
    def scale(self):
        return self._prop_manager.ticks
    @scale.setter
    def scale(self, value):
        self._update_copy_opt('scale', value)
        self._prop_manager.scale = value

    @property
    def aspect_ratio(self):
        return self._prop_manager.aspect_ratio
    @aspect_ratio.setter
    def aspect_ratio(self, value):
        self._update_copy_opt('aspect_ratio', value)
        self._prop_manager.aspect_ratio = value

    @property
    def image_size(self):
        return self._prop_manager.image_size
    @image_size.setter
    def image_size(self, value):
        # print(">>>", value)
        self._update_copy_opt('image_size', value)
        self._prop_manager.image_size = value

    @property
    def padding(self):
        return self._prop_manager.padding
    @padding.setter
    def padding(self, value):
        self._update_copy_opt('padding', value)
        self._prop_manager.padding = value
    @property
    def padding_left(self):
        return self._prop_manager.padding_left
    @padding_left.setter
    def padding_left(self, value):
        self._update_copy_opt('padding_left', value)
        self._prop_manager.padding_left = value
    @property
    def padding_right(self):
        return self._prop_manager.padding_right
    @padding_right.setter
    def padding_right(self, value):
        self._update_copy_opt('padding_right', value)
        self._prop_manager.padding_right = value
    @property
    def padding_top(self):
        return self._prop_manager.padding_top
    @padding_top.setter
    def padding_top(self, value):
        self._update_copy_opt('padding_top', value)
        self._prop_manager.padding_top = value
    @property
    def padding_bottom(self):
        return self._prop_manager.padding_bottom
    @padding_bottom.setter
    def padding_bottom(self, value):
        self._update_copy_opt('padding_bottom', value)
        self._prop_manager.padding_bottom = value

    @property
    def spacings(self):
        return self._prop_manager.spacings
    @spacings.setter
    def spacings(self, value):
        self._update_copy_opt('spacings', value)
        self._prop_manager.spacings = value

    @property
    def background(self):
        return self._prop_manager.background
    @background.setter
    def background(self, value):
        self._update_copy_opt('background', value)
        self._prop_manager.background = value

    @property
    def colorbar(self):
        return self._prop_manager.colorbar
    @colorbar.setter
    def colorbar(self, value):
        self._update_copy_opt('colorbar', value)
        self._prop_manager.colorbar = value

    def prep_show(self):
        super().prep_show()
        for c in self.children:
            c.prep_show()
        if self.plot_legend or any(c.plot_legend for c in self.children):
            ls = self.legend_style if self.legend_style is not None else {}
            self.axes.legend(**ls)

########################################################################################################################
#
#                                               Graphics3D
#
class Graphics3D(Graphics):
    """Extends the standard matplotlib 3D plotting to use all the Graphics extensions"""

    opt_keys = GraphicsBase.opt_keys | {'view_settings'}

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
                 view_settings=None,
                 backend=Backends.MPL,
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
            view_settings=view_settings,
            prop_manager=GraphicsPropertyManager3D,
            **kwargs
        )

    def set_options(self,
                    view_settings=None,
                    **parent_opts
                    ):

        super().set_options(**parent_opts)

        opts = (
            ('view_settings', view_settings),
        )
        for oname, oval in opts:
            oval = self._get_def_opt(oname, oval, {})
            if oval is not None:
                setattr(self, oname, oval)

    def load_mpl(self):
        if not self._mpl_loaded:
            from mpl_toolkits.mplot3d import Axes3D
            super().load_mpl()

    @staticmethod
    def _subplot_init(*args, backend = Backends.MPL, mpl_backend=None, graphics=None, **kw):
        if backend == Backends.VTK:
            from .VTKInterface import VTKWindow
            window = VTKWindow()
            return window, window
        else:
            with graphics.pyplot as plt:
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
            figure, axes = self._subplot_init(*args, backend=self._backend, mpl_backend=self.mpl_backend, **kw)
        elif isinstance(figure, GraphicsBase):
            axes = figure.axes
            figure = figure.figure

        if axes is None:
            if self._backend == Backends.MPL:
                axes = figure
            else:
                axes = figure.add_subplot(1, 1, 1, projection='3d')

        return figure, axes

    @property
    def plot_label(self):
        return self._prop_manager.plot_label
    @plot_label.setter
    def plot_label(self, value):
        self._prop_manager.plot_label = value

    @property
    def plot_legend(self):
        return self._prop_manager.plot_legend
    @plot_legend.setter
    def plot_legend(self, value):
        self._prop_manager.plot_legend = value

    @property
    def axes_labels(self):
        return self._prop_manager.axes_labels
    @axes_labels.setter
    def axes_labels(self, value):
        self._prop_manager.axes_labels = value

    @property
    def frame(self):
        return self._prop_manager.frame
    @frame.setter
    def frame(self, value):
        self._prop_manager.frame = value

    @property
    def plot_range(self):
        return self._prop_manager.plot_range
    @plot_range.setter
    def plot_range(self, value):
        self._prop_manager.plot_range = value

    @property
    def ticks(self):
        return self._prop_manager.ticks
    @ticks.setter
    def ticks(self, value):
        self._prop_manager.ticks = value

    @property
    def ticks_style(self):
        return self._prop_manager.ticks_style
    @ticks_style.setter
    def ticks_style(self, value):
        self._prop_manager.ticks_style = value

    @property
    def ticks_label_style(self):
        return self._prop_manager.ticks_label_style
    @ticks_label_style.setter
    def ticks_label_style(self, value):
        self._prop_manager.ticks_label_style = value

    @property
    def scale(self):
        return self._prop_manager.ticks
    @scale.setter
    def scale(self, value):
        self._prop_manager.scale = value

    @property
    def aspect_ratio(self):
        return self._prop_manager.aspect_ratio
    @aspect_ratio.setter
    def aspect_ratio(self, value):
        self._prop_manager.aspect_ratio = value

    @property
    def image_size(self):
        return self._prop_manager.image_size
    @image_size.setter
    def image_size(self, value):
        self._prop_manager.image_size = value

    @property
    def padding(self):
        return self._prop_manager.padding
    @padding.setter
    def padding(self, value):
        self._prop_manager.padding = value

    @property
    def background(self):
        return self._prop_manager.background
    @background.setter
    def background(self, value):
        self._prop_manager.background = value

    @property
    def colorbar(self):
        return self._prop_manager.colorbar
    @colorbar.setter
    def colorbar(self, value):
        self._prop_manager.colorbar = value

    # set plot ranges
    @property
    def ticks(self):
        return self._ticks
    @ticks.setter
    def ticks(self, value):
        self._prop_manager.ticks = value

    @property
    def box_ratios(self):
        return self._prop_manager.box_ratios
    @box_ratios.setter
    def box_ratios(self, value):
        self._prop_manager.box_ratios = value

    @property
    def view_settings(self):
        return self._prop_manager.view_settings
    @view_settings.setter
    def view_settings(self, value):
        self._prop_manager.view_settings = value

    @property
    def aspect_ratio(self):
        return self._aspect_ratio
    @aspect_ratio.setter
    def aspect_ratio(self, ar):
        pass
        # if isinstance(ar, (float, int)):
        #     a, b = self.plot_range
        #     cur_ar = (b[1] - b[0]) / (a[1] - a[0])
        #     targ_ar = ar / cur_ar
        #     self.axes.set_aspect(targ_ar)
        # elif isinstance(ar, str):
        #     self.axes.set_aspect(ar)
        # else:
        #     self.axes.set_aspect(ar[0], **ar[1])

########################################################################################################################
#
#                                               GraphicsGrid
#
class GraphicsGrid(GraphicsBase):
    """
    A class for easily building sophisticated multi-panel figures.
    Robustification work still needs to be done, but the core interface is there.
    Supports themes & direct, easy access to the panels, among other things.
    Builds off of `GraphicsBase`.
    """
    default_style = dict(
        theme='mccoy',
        spacings=(50, 0),
        padding=((50, 10), (50, 10))
    )
    def __init__(self,
                 *args,
                 nrows=2, ncols=2,
                 graphics_class=Graphics,
                 figure=None,
                 axes=None,
                 subplot_kw=None,
                 _subplot_init=None,
                 mpl_backend = None,
                 subimage_size=(200, 200),
                 subimage_aspect_ratio='auto',
                 padding=None,
                 spacings=None,
                 **opts
                 ):

        # if mpl_backend is None and platform.system() == "Darwin":
        #     mpl_backend = "TkAgg"
        self.mpl_backend = mpl_backend
        if subplot_kw is None:
            subplot_kw={}
        else:
            subplot_kw={'subplot_kw':subplot_kw}
        subplot_kw.update(dict(
            nrows=nrows, ncols=ncols,
            graphics_class=graphics_class,
            _subplot_init=graphics_class._subplot_init if _subplot_init is None else _subplot_init,
            subimage_size=subimage_size,
            subimage_aspect_ratio=subimage_aspect_ratio,
            padding=padding,
            spacings=spacings
        ))
        super().__init__(
            figure=figure, axes=axes,
            graphics_class=graphics_class,
            subplot_kw=subplot_kw,
            padding=padding,
            spacings=spacings,
            **opts
        )
        self.shape = (nrows, ncols)
        self._colorbar_axis = None # necessary hack for only GraphicsGrid

    def _init_suplots(self,
                      figure, axes, *args,
                      nrows=None, ncols=None, graphics_class=None,
                      subplot_kw=None, _subplot_init=None,
                      fig_kw=None, mpl_backend = None,
                      padding = None, spacings=None,
                      subimage_size=None, subimage_aspect_ratio=None,
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
        import matplotlib.figure, matplotlib.axes
        if mpl_backend is not None:
            import matplotlib
            matplotlib.use(mpl_backend)

        if 'image_size' not in kw:
            kw['image_size'] = subimage_size
        else:
            subimage_size = kw['image_size']
        if figure is None:
            padding = self._get_def_opt('padding', padding, {})
            subimage_aspect_ratio = self._get_def_opt('padding', subimage_aspect_ratio, {})
            spacings = self._get_def_opt('spacings', spacings, {})
            if subplot_kw is None:
                subplot_kw = {}
            if fig_kw is None:
                fig_kw = {}
            if 'figsize' not in fig_kw:
                try:
                    dw, dh = subimage_size
                except (ValueError, TypeError):
                    dw = dh = subimage_size
                w = ncols * dw
                h = nrows * dh
                if padding is not None:
                    pw, ph = padding
                    try:
                        pwx, pwy = pw
                    except (TypeError, ValueError):
                        pwx = pwy = pw
                    try:
                        phx, phy = ph
                    except (TypeError, ValueError):
                        phx = phy = ph

                    w += pwx + pwy
                    h += phx + phy
                if spacings is not None:
                    sw, sh = spacings
                    w += (ncols-1) * sw
                    h += (nrows-1) * sh
                fig_kw['figsize'] = (w/72., h/72.)

            figure, axes = _subplot_init(*args, nrows = nrows, ncols=ncols, subplot_kw=subplot_kw, figure=self, **fig_kw)

            if isinstance(axes, matplotlib.axes.Axes):
                axes = [[axes]]
            elif isinstance(axes[0], matplotlib.axes.Axes):
                if ncols == 1:
                    axes = [[ax] for ax in axes]
                else:
                    axes = [axes]
            if 'aspect_ratio' not in kw:
                kw['aspect_ratio'] = subimage_aspect_ratio
            for i in range(nrows):
                for j in range(ncols):
                    axes[i][j] = graphics_class(figure=figure, axes=axes[i][j], managed=True, **kw)
        elif isinstance(figure, GraphicsGrid):
            axes = figure.axes  # type: matplotlib.axes.Axes
            figure = figure.figure  # type: matplotlib.figure.Figure

        if axes is None:
            axes = [
                graphics_class(
                    figure.add_subplot(nrows, ncols, i),
                    managed=True,
                    **kw
                ) for i in range(nrows * ncols)
            ]

        return figure, axes

    def set_options(self,
                    padding=None,
                    spacings=None,
                    background=None,
                    colorbar=None,
                    figure_label=None,
                    **parent_opts
                    ):

        super().set_options(**parent_opts)

        self.image_size = None
        opts = (
            ('figure_label', figure_label),
            ('padding', padding),
            ('spacings', spacings),
            ('background', background),
            ('colorbar', colorbar)
        )
        for oname, oval in opts:
            oval = self._get_def_opt(oname, oval, {})
            if oval is not None:
                setattr(self, oname, oval)

    def __iter__(self):
        import itertools as ip
        return ip.chain(*self.axes)
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

    # def __getattr__(self, item):
    #     reraise_error = None
    #     figure = object.__getattribute__(self, 'figure')
    #     try:
    #         meth = getattr(figure, item)
    #     except AttributeError as e:
    #         if 'Figure' not in e.args[0] or item not in e.args[0]:
    #             reraise_error = e
    #         else:
    #             reraise_error = AttributeError("'{}' object has no attribute '{}'".format(
    #                 type(self).__name__,
    #                 item
    #             ))
    #
    #     if reraise_error is not None:
    #         raise reraise_error
    #
    #     return meth

    # set size
    def calc_image_size(self):
        w=0; h=0
        for l in self.axes:
            mh = 0
            mw = 0
            for f in l:
                wh = f.image_size
                if wh is not None:
                    mw += wh[0]
                    if wh[1] > mh:
                        mh = wh[1]
            h += mh
            if w < mw:
                w = mw

        spacings = self.spacings
        if spacings is not None:
            sw, sh = spacings
            w += (self.shape[1] - 1) * sw
            h += (self.shape[0] - 1) * sh

        pad = self.padding
        if pad is not None:
            pw, ph = pad
            w += pw[0] + pw[1]
            h += ph[0] + ph[1]
        return w, h

    @property
    def image_size(self):
        s = self.calc_image_size()
        self._prop_manager.image_size = s
        return s
    @image_size.setter
    def image_size(self, value):
        s = self.calc_image_size()
        self._prop_manager.image_size = s

    @property
    def figure_label(self):
        return self._prop_manager.figure_label
    @figure_label.setter
    def figure_label(self, value):
        self._prop_manager.figure_label = value
    @property
    def padding(self):
        return self._prop_manager.padding
    @padding.setter
    def padding(self, value):
        self._prop_manager.padding = value
    @property
    def padding_left(self):
        return self._prop_manager.padding_left
    @padding_left.setter
    def padding_left(self, value):
        self._prop_manager.padding_left = value
    @property
    def padding_right(self):
        return self._prop_manager.padding_right
    @padding_right.setter
    def padding_right(self, value):
        self._prop_manager.padding_right = value
    @property
    def padding_top(self):
        return self._prop_manager.padding_top
    @padding_top.setter
    def padding_top(self, value):
        self._prop_manager.padding_top = value
    @property
    def padding_bottom(self):
        return self._prop_manager.padding_bottom
    @padding_bottom.setter
    def padding_bottom(self, value):
        self._prop_manager.padding_bottom = value

    @property
    def spacings(self):
        return self._prop_manager.spacings
    @spacings.setter
    def spacings(self, value):
        self._prop_manager.spacings = value

    @property
    def background(self):
        return self._prop_manager.background
    @background.setter
    def background(self, value):
        self._prop_manager.background = value

    @property
    def colorbar(self):
        return self._prop_manager.colorbar
    @colorbar.setter
    def colorbar(self, value):
        self._prop_manager.colorbar = value

    def prep_show(self):
        self.image_size = None
        if self.spacings is not None:
            self.spacings = self.spacings
        if self.padding is not None:
            self.padding = self.padding
        if self.tighten:
            self.figure.tight_layout()

    def show(self, **kwargs):
        for f in self:
            if isinstance(f, GraphicsBase):
                f.prep_show()
        super().show(**kwargs)

