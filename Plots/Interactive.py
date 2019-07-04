"""
Convenience classes for hooking into the matplotlib animation framework
"""
from .Graphics import Graphics

class EventHandler:
    def __init__(self,
                 figure,
                 on_click = None,
                 on_release = None,
                 on_draw = None,
                 on_key_press = None,
                 on_key_release = None,
                 on_move = None,
                 on_select = None,
                 on_resize = None,
                 on_scroll = None,
                 on_figure_entered = None,
                 on_figure_left = None,
                 on_axes_entered = None,
                 on_axes_left = None
                 ):
        """Creates an EventHandler on a Figure that handles most interactivity stuff

        :param figure:
        :type figure: GraphicsBase
        :param on_click:
        :type on_click:
        :param on_release:
        :type on_release:
        :param on_draw:
        :type on_draw:
        :param on_key_press:
        :type on_key_press:
        :param on_key_release:
        :type on_key_release:
        :param on_move:
        :type on_move:
        :param on_select:
        :type on_select:
        :param on_resize:
        :type on_resize:
        :param on_scroll:
        :type on_scroll:
        :param on_figure_entered:
        :type on_figure_entered:
        :param on_figure_left:
        :type on_figure_left:
        :param on_axes_entered:
        :type on_axes_entered:
        :param on_axes_left:
        :type on_axes_left:
        """

        self.figure = figure

        if not isinstance(on_click, self.Event):
            on_click = self.ButtonPressedEvent(on_click)
        if not isinstance(on_release, self.Event):
            on_release = self.ButtonReleasedEvent(on_release)
        if not isinstance(on_draw, self.Event):
            on_draw = self.DrawEvent(on_draw)
        if not isinstance(on_key_press, self.Event):
            on_key_press = self.KeyPressedEvent(on_key_press)
        if not isinstance(on_key_release, self.Event):
            on_key_release = self.KeyReleasedEvent(on_key_release)
        if not isinstance(on_move, self.Event):
            on_move = self.MoveEvent(on_move)
        if not isinstance(on_select, self.Event):
            on_select = self.SelectEvent(on_select)
        if not isinstance(on_scroll, self.Event):
            on_scroll = self.ScrollEvent(on_scroll)
        if not isinstance(on_figure_entered, self.Event):
            on_figure_entered = self.FigureEnterEvent(on_figure_entered)
        if not isinstance(on_figure_left, self.Event):
            on_figure_left = self.FigureLeaveEvent(on_figure_left)
        if not isinstance(on_axes_entered, self.Event):
            on_axes_entered = self.AxesEnterEvent(on_axes_entered)
        if not isinstance(on_axes_left, self.Event):
            on_axes_left = self.AxesEnterEvent(on_axes_left)

        self._handles = dict(
            button_press_event = on_click,
            button_release_event =on_release,
            draw_event = on_draw,
            key_press_event = on_key_press,
            key_release_event = on_key_release,
            motion_notify_event = on_move,
            pick_event = on_select,
            resize_event = on_resize,
            scroll_event = on_scroll,
            figure_enter_event = on_figure_entered,
            figure_leave_event = on_figure_left,
            axes_enter_event = on_axes_entered,
            axes_leave_event = on_axes_left
        )

        self.bind(**self._handles)

    def bind(self, **handlers):
        self._handles.update(**handlers)
        for handle, handler in handlers.items():
            self.figure.canvas.mpl_connect(handle, handler)

    class Event:
        def __init__(self,
                     event_handler,
                     handler,
                     filter = None,
                     update = True,
                     name = None
                     ):
            self.filter = filter
            self.handler = handler
            self.parent = event_handler
            self.update = update,
            self.name = name

        def handle_event(self, event):
            if self.filter is not None:
                handle = self.filter(event)
            else:
                handle = True
            if handle:
                res = self.handler(event)
                if (res is not False) and (res is not None or self.update):
                    self.parent.figure.canvas.draw()
            else:
                res = None
            return res

    def ButtonPressedEvent(self, handler, **kw):
        kw = dict({'name' : 'ButtonPressed'}, **kw)
        return self.Event(self, handler, **kw)
    def ButtonReleasedEvent(self, handler, **kw):
        kw = dict({'name' : 'ButtonReleased'}, **kw)
        return self.Event(self, handler, **kw)
    def DrawEvent(self, handler, **kw):
        kw = dict({'name' : 'Draw'}, **kw)
        return self.Event(self, handler, **kw)
    def KeyPressedEvent(self, handler, **kw):
        if isinstance(handler, dict):
            filt = lambda e, h=handler: e.key in h
            handler = lambda e, h = handler: h[e.key](e)
        else:
            filt = None
        kw = dict({'name' : 'KeyPressed', 'filter':filt}, **kw)
        return self.Event(self, handler, **kw)
    def KeyReleasedEvent(self, handler, **kw):
        if isinstance(handler, dict):
            filt = lambda e, h=handler: e.key in h
            handler = lambda e, h = handler: h[e.key](e)
        else:
            filt = None
        kw = dict({'name' : 'KeyReleased', 'filter':filt}, **kw)
        return self.Event(self, handler, **kw)
    def MoveEvent(self, handler, **kw):
        kw = dict({'name' : 'Move'}, **kw)
        return self.Event(self, handler, **kw)
    def SelectEvent(self, handler, **kw):
        kw = dict({'name' : 'Select'}, **kw)
        return self.Event(self, handler, **kw)
    def ScrollEvent(self, handler, **kw):
        kw = dict({'name' : 'Scroll'}, **kw)
        return self.Event(self, handler, **kw)
    def FigureEnterEvent(self, handler, **kw):
        kw = dict({'name' : 'FigureEnter'}, **kw)
        return self.Event(self, handler, **kw)
    def FigureLeaveEvent(self, handler, **kw):
        kw = dict({'name' : 'FigureLeave'}, **kw)
        return self.Event(self, handler, **kw)
    def AxesEnterEvent(self, handler, **kw):
        kw = dict({'name' : 'AxesEnter'}, **kw)
        return self.Event(self, handler, **kw)
    def AxesLeaveEvent(self, handler, **kw):
        kw = dict({'name' : 'AxesLeave'}, **kw)
        return self.Event(self, handler, **kw)
