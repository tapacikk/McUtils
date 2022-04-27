
__all__ = [
    "App",
    "Manipulator"
]

import types

from ..JHTML import JHTML#, HTML, ActiveHTMLWrapper
from ..JHTML.WidgetTools import JupyterAPIs
from .Interfaces import *
from .Controls import Control, FunctionDisplay
from .Variables import WidgetControl, InterfaceVars


__reload_hook__ = ['..JHTML', '.Interfaces', '.Controls', '.Variables']

# class FunctionDisplay:
#     def __init__(self, func, vars):
#         self.func = func
#         self.vars = vars
#         self.output = JupyterAPIs.get_widgets_api().Output()
#         self.update = self.update # a weird kluge to prevent a weakref issue...
#     def display(self):
#         self.output.clear_output()
#         with self.output:
#             res = self.func(**{x.name: x.value for x in self.vars})
#             if res is not None:
#                 JupyterAPIs.get_display_api().display(res)
#     def update(self, *ignored_settings):
#         return self.display()
#     def to_widget(self):
#         for v in self.vars:
#             v.callbacks.add(self.update)
#         return self.output
class Manipulator(WidgetInterface):
    def __init__(self, func, *controls):
        self.controls = [self.canonicalize_control(c) for c in controls]
        vars = [c.var for c in self.controls]
        self.output = FunctionDisplay(func, vars)
    @classmethod
    def canonicalize_control(cls, settings):
        if isinstance(settings, (WidgetControl, Control)):
            return settings
        else:
            return WidgetControl(settings[0], **settings[1])
    def to_widget(self):
        widgets = JupyterAPIs.get_widgets_api()
        elems = [c.to_widget() for c in self.controls] + [self.output.to_widget()]
        return widgets.VBox(elems)
    def initialize(self):
        self.output.update()

class App(Component):
    """
    Provides a framework for making Jupyter Apps with the
    elements built out in the Interfaces package
    """
    def __init__(self,
                 body=None,
                 header=None,
                 footer=None,
                 sidebar=None,
                 toolbar=None,
                 layout='grid',
                 cls='border',
                 **attrs
                 ):
        super().__init__()
        self.vars = InterfaceVars()
        self.body = body
        self.header = header
        self.footer = footer
        self.sidebar = sidebar
        self.toolbar = toolbar
        self.layout = layout
        self.cls = cls
        self.attrs = attrs

    @classmethod
    def construct_navbar_item(cls, item):
        if isinstance(item, tuple):
            k,v = item
            if isinstance(v, tuple):
                item = Dropdown(k, v)
        return item
    @classmethod
    def construct_header(cls, header):
        if isinstance(header, (list, tuple)):
            header = Navbar(
                [cls.construct_navbar_item(h) for h in header],
                cls=['navbar-dark', 'bg-dark', 'border-bottom']
            )
        # elif isinstance(header, str):
        #     header = ...
        # elif isinstance(header, (HTML.XMLElement, ActiveHTMLWrapper)):
        #     ...
        return header
    @classmethod
    def construct_footer(cls, footer):
        if isinstance(footer, (list, tuple)):
            footer = Navbar(footer,
                cls=['navbar-light', 'bg-light', 'border-bottom']
            )
        return footer

    @classmethod
    def construct_sidebar_item(cls, item):
        if isinstance(item, tuple):
            k,v = item
            if isinstance(v, tuple):
                item = Opener(k, v)
        return item
    @classmethod
    def construct_sidebar(cls, sidebar):
        if isinstance(sidebar, (list, tuple)):
            sidebar = Sidebar(
                [cls.construct_sidebar_item(h) for h in sidebar],
                cls=['bg-light', 'border-end', 'h-100']
            )
        return sidebar

    @classmethod
    def construct_toolbar_item(cls, item):
        if isinstance(item, dict):
            item = item.copy()
            var = item['var']
            del item['var']
            item = Control.construct(var, **item)
        return item
    @classmethod
    def construct_toolbar(cls, toolbar):
        if isinstance(toolbar, (list, tuple)):
            if isinstance(toolbar[0], (list, tuple)):
                toolbar = Grid(
                    [[cls.construct_toolbar_item(h) for h in row] for row in toolbar],
                    cls=['form-check', 'bg-light', 'border-bottom']
                )
            else:
                toolbar = JHTML.Div(
                    [cls.construct_toolbar_item(h) for h in toolbar],
                    cls=['form-check', 'bg-light', 'border-bottom']
                )
        return toolbar

    def wrap_body(self, fn, **styles):
        # @functools.wraps(fn)
        # def wrapper(event=None, pane=None, **vars):
        #     return fn(event=event, pane=pane, **vars)
        return FunctionDisplay(fn, self.vars)
    def construct_body_item(self, item):
        if isinstance(item, (types.MethodType, types.FunctionType, types.LambdaType)):
            item = self.wrap_body(item)
        elif not isinstance(item, str):
            try:
                item, styles = item
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(item, (types.MethodType, types.FunctionType, types.LambdaType)):
                    print(item)
                    item = self.wrap_body(item, **styles)
                else:
                    item = JHTML.Span(item, **styles)
        return item
    def construct_body(cls, body):
        if isinstance(body, dict):
            body = Tabs({k:cls.construct_body_item(b) for k,b in body.items()})
        else:
            body = cls.construct_body_item(body)
        return body

    def construct_layout(self):
        with self.vars:
            if self.layout == 'grid':
                elements = []
                nrows = 0
                ncols = 0

                # if self.header is not None:
                #     nrows += 1
                if self.sidebar is not None:
                    ncols += 1
                if self.toolbar is not None:
                    nrows += 1
                    ncols += 1
                if self.body is not None:
                    nrows += 1
                    ncols = min(ncols+1, 2)
                # if self.footer is not None:
                #     nrows += 1

                column_width = []
                row_height = []
                if self.header is not None:
                    header = []
                    elem = self.construct_header(self.header)
                    header.append(Grid.Item(elem, col_span=ncols))
                    elements.append(header)
                    row_height.append('auto')

                body = []
                if self.sidebar is not None:
                    elem = self.construct_sidebar(self.sidebar)
                    body.append(Grid.Item(elem, row_span=nrows))
                    column_width.append('auto')
                if self.toolbar is not None:
                    body.append(self.construct_toolbar(self.toolbar))
                    elements.append(body)
                    body = [None]
                    row_height.append('auto')
                body.append(self.construct_body(self.body))
                column_width.append('1fr')
                row_height.append('1fr')
                elements.append(body)

                if self.footer is not None:
                    footer = []
                    elem = self.construct_footer(self.footer)
                    footer.append(Grid.Item(elem, col_span=ncols))
                    elements.append(footer)
                    row_height.append('auto')
                return Grid(
                    elements,
                    column_width=column_width,
                    row_height=row_height,
                    cls=self.cls
                )
            else:
                raise NotImplementedError("ugh")
    def to_jhtml(self):
        return self.construct_layout().to_jhtml()