
__all__ = [
    "App",
    "FunctionDisplay",
    "Manipulator"
]

from ..JHTML import JHTML#, HTML, ActiveHTMLWrapper
from ..JHTML.WidgetTools import JupyterAPIs
from .Interfaces import *
from .Controls import Control
from .Variables import WidgetControl


__reload_hook__ = ['..JHTML', '.Interfaces', '.Controls', '.Variables']

class FunctionDisplay:
    def __init__(self, func, vars):
        self.func = func
        self.vars = vars
        self.output = JupyterAPIs.get_widgets_api().Output()
        self.update = self.update # a weird kluge to prevent a weakref issue...
    def display(self):
        self.output.clear_output()
        with self.output:
            res = self.func(**{x.name: x.value for x in self.vars})
            if res is not None:
                JupyterAPIs.get_display_api().display(res)
    def update(self, *ignored_settings):
        return self.display()
    def to_widget(self):
        for v in self.vars:
            v.callbacks.add(self.update)
        return self.output
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
        self.body = body
        self.header = header
        self.footer = footer
        self.sidebar = sidebar
        self.toolbar = toolbar
        self.layout = layout
        self.attrs = attrs

    @classmethod
    def construct_menu_item(cls, item):
        if isinstance(item, tuple):
            k,v = item
            if isinstance(v, tuple):
                item = Dropdown(k, v)
        return item

    @classmethod
    def construct_header(cls, header):
        if isinstance(header, (list, tuple)):
            header = Navbar([cls.construct_menu_item(h) for h in header])
        # elif isinstance(header, str):
        #     header = ...
        # elif isinstance(header, (HTML.XMLElement, ActiveHTMLWrapper)):
        #     ...
        return header

    # @classmethod
    # def construct_header(cls, header):
    #     ...

    def construct_layout(self):
        if self.layout == 'grid':
            elements = []
            column_width = []
            row_height = []
            if self.header is not None:
                header = []
                header.append(self.construct_header(self.header))
                elements.append(header)
                row_height.append('auto')
            body = []
            if self.sidebar is not None:
                body.append(self.construct_sidebar(self.sidebar))
            if self.toolbar is not None:
                body.append(self.construct_toolbar(self.toolbar))
            if self.body is not None:
                body.append(self.construct_body(self.body))

            if self.footer is not None:
                footer = []
                footer.append(self.construct_footer(self.footer))
                elements.append(footer)

            elements.append(self.body)
            return Grid(
                elements,
                column_width=['auto', '1fr'],
                row_height=['auto', '1fr'],
                cls='border'
            )
        else:
            raise NotImplementedError("ugh")
    def to_jhtml(self):
        return self.construct_layout().to_jhtml()