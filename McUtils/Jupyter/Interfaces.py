
import abc, weakref
from .JHMTL import JHTML
from .WidgetTools import JupyterAPIs, frozendict
from .Controls import Control, Var

__all__ = [
    "Manipulator",
    "Grid",
    "InputField",
    "StringField",
    "Checkbox",
    "Slider",
    "RadioButton",
    "TextArea",
    "Select",
    "DropdownMenu"
]
__reload_hook__ = [".JHTML", ".WidgetTools", ".Controls"]

# class OutputPane(InterfaceElement):
#
#     def __init__(self, autoclear=False):
#         self.autoclear = autoclear
#     def to_jhtml(self):
#         return self.


# class Dashboard:
#
#     class Sidebar:
#         ...
#
#     class Header:
#         ...
#
#     class DisplayPane:
#         ...

class WidgetInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_widget(self):
        ...
    # @abc.abstractmethod
    def initialize(self):
        ...
    def _ipython_display_(self):
        JupyterAPIs.get_display_api().display(self.to_widget())
        self.initialize()
    def display(self):
        self._ipython_display_()
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
        if isinstance(settings, (Control, JHTMLControl)):
            return settings
        else:
            return Control(settings[0], **settings[1])
    def to_widget(self):
        widgets = JupyterAPIs.get_widgets_api()
        elems = [c.to_widget() for c in self.controls] + [self.output.to_widget()]
        return widgets.VBox(elems)
    def initialize(self):
        self.output.update()

class Component(WidgetInterface):
    """
    Provides an abstract base class for an interface element
    to allow for the easy construction of interesting interfaces
    """
    def __init__(self, dynamic=True, **attrs):
        self._parents = weakref.WeakSet()
        self._widget_cache = None
        attrs['dynamic'] = dynamic
        self._attrs = attrs
    @property
    def attrs(self):
        return frozendict(self._attrs)
    @attrs.setter
    def attrs(self, value):
        self._attrs = value
    def __getitem__(self, item):
        return self._attrs[item]
    def __setitem__(self, key, value):
        self._attrs[key] = value
        self.invalidate_cache()

    @abc.abstractmethod
    def to_jhtml(self):
        ...
    def to_widget(self, parent=None):
        if parent is not None:
            self._parents.add(parent)
        if self._widget_cache is None:
            self._widget_cache = self.to_jhtml()
            # self._widget_cache.to_widget.observe(self.set_value, )
        return self._widget_cache
    def mutate(self, fn):
        fn(self)
        self.invalidate_cache()
    def invalidate_cache(self):
        self._widget_cache = None
        for w in self._parents:
            w.invalidate_cache()

class WrapperComponent(Component):
    wrapper = JHTML.Div
    wrapper_classes = []
    def __init__(self, items, cls=None, **attrs):
        super().__init__(**attrs)
        self.items = items
        self.wrapper_classes = self.wrapper_classes + JHTML.manage_cls(cls)
    def wrap_items(self, items):
        return self.wrapper(*items, cls=self.wrapper_classes, **self.attrs)
    def to_jhtml(self, parent=None):
        return self.wrap_items(self.items)
class Container(WrapperComponent):
    item = JHTML.Span
    item_classes = []
    def __init__(self, items, item_attrs=None, **attrs):
        super().__init__(None, **attrs)
        self._items = items
        if 'cls' in item_attrs:
            item_attrs = item_attrs.copy()
            self.item_classes = self.item_classes + JHTML.manage_cls(item_attrs['cls'])
            del item_attrs['cls']
        self.item_attrs = item_attrs
    @property
    def items(self):
        return [self.create_item(i) for i in self._items]
    def _create_dict_item(self, body=None, **extra):
        return self.item(body, cls=self.item_classes, **dict(self.item_attrs, **extra))
    def _create_base_item(self, body):
        return self.item(body, cls=self.item_classes, **self.item_attrs)
    def create_item(self, i):
        if isinstance(i, dict):
            return self._create_dict_item(**i)
        else:
            return self._create_base_item(i)

#region Menus

class MenuComponent(Container):
    def __init__(self, items, item_attrs=None, cls=None, **attrs):
        super().__init__(items, item_attrs=item_attrs, cls=cls, **attrs)
        self._item_map = {}
    def create_item(self, item):
        item = super().create_item(item)
        if hasattr(item, 'id'):
            self._item_map[item.id] = item
        elif 'id' in item.attrs:
            self._item_map[item.attrs['id']] = item
        return item
class ListGroup(MenuComponent):
    item_classes = ['list-group-item', 'list-group-item-action']
    wrapper_classes = ['list-group', 'list-group-flush']
class DropdownList(MenuComponent):
    wrapper = JHTML.Ul
    wrapper_classes = ['dropdown-menu']
    item = JHTML.Compound(JHTML.Li, JHTML.Anchor)
    item_classes = ['dropdown-item']
    def canonicalize_item(self, i):
        if not isinstance(i, dict):
            i = {'body':i}
        if 'action' in i:
            i = i.copy()
            i['event_handlers'] = {'click':i['action']}
            del i['action']
        return i
class DropdownMenu(MenuComponent):
    wrapper_classes = ['dropdown']
    toggle_classes = ['dropdown-toggle']
    toggle = JHTML.Styled(JHTML.Bootstrap.Button, data_bs_toggle='dropdown')
    def __init__(self, header, actions, toggle_attrs=None, **attrs):
        self.header = header
        self.toggle_attrs = {} if toggle_attrs is None else toggle_attrs
        if 'cls' in self.toggle_attrs:
            self.toggle_attrs = self.toggle_attrs.copy()
            self.toggle_classes = self.toggle_classes + JHTML.manage_cls(self.toggle_attrs['cls'])
            del self.toggle_attrs['cls']
        self.dlist = DropdownList(actions, **attrs)
        super().__init__([], **attrs)
    def wrap_items(self, items):
        items = [
            self.toggle(self.header, cls=self.toggle_classes, **self.toggle_attrs),
            self.dlist
        ]
        return super().wrap_items(items)

#endregion

#region JHTML Controls
class JHTMLControl(Component):
    layout_orientation = 'row'
    def __init__(self, var):
        self.var = Var(var)
        self._widget_cache = None
        super().__init__()
    def to_widget(self, parent=None):
        needs_link = self._widget_cache is None
        widg = super().to_widget(parent=parent)
        if needs_link:
            self.var.link(self)
        return widg
    @abc.abstractmethod
    def set_value(self):
        ...
    @abc.abstractmethod
    def get_value(self):
        ...
    @property
    def value(self):
        return self.get_value()
    @value.setter
    def value(self, v):
        self.var.set_value(v, caller=self)
        self.set_value()
    def observe(self, fn, names=None):
        if self._widget_cache is None:
            raise ValueError("not initialized")
        return self._widget_cache.to_widget().observe(fn, names=names)

class ValueWidget(JHTMLControl):
    def __init__(self, var, value=None):
        super().__init__(var)
        if value is not None:
            self.var.value = value
        if self.var.value is None:
            self.var.value = ""
    def get_value(self):
        if self._widget_cache is not None:
            val = self._widget_cache.value
            return "" if val is None else val
    def set_value(self):
        if self._widget_cache is not None:
            self._widget_cache.value = self.var.value
    def update(self, e):
        if self._widget_cache is not None:
            self.var.value = self._widget_cache.value
class InputField(ValueWidget):
    def __init__(self, var, value=None, tag='input', track_value=True, continuous_update=False, **attrs):
        super().__init__(var, value=value)
        attrs['tag'] = tag
        attrs['value'] = value
        attrs['track_value'] = track_value
        attrs['continuous_update'] = continuous_update
        self.attrs = attrs
    def to_jhtml(self):
        field = JHTML.Input(**self.attrs)
        return field
class StringField(InputField):
    def __init__(self, var, type='text', **attrs):
        super().__init__(var, type=type, **attrs)
class Slider(InputField):
    def __init__(self, var, type='range', **attrs):
        super().__init__(var, type=type, **attrs)
class Checkbox(InputField):
    def __init__(self, var, type='checkbox', **attrs):
        super().__init__(var, type=type, **attrs)
    def get_value(self):
        if self._widget_cache is not None:
            val = self._widget_cache.value
            return isinstance(val, str) and val == "true"
    def set_value(self):
        if self._widget_cache is not None:
            if self.var.value:
                self._widget_cache.value = 'true'
            else:
                self._widget_cache.value = 'false'
class RadioButton(InputField):
    def __init__(self, var, type='radio', **attrs):
        super().__init__(var, type=type, **attrs)
class TextArea(InputField):
    def __init__(self, var, tag='textarea', **attrs):
        super().__init__(var, tag=tag, **attrs)
    def to_jhtml(self):
        field = JHTML.Textarea(**self.attrs)
        return field
class ChangeTracker(ValueWidget):
    base = None
    def __init__(self, var, base=None, value=None, track_value=True, continuous_update=False, **attrs):
        super().__init__(var, value=value)
        base = self.base if base is None else base
        self.base = getattr(JHTML, base) if isinstance(base, str) else base
        attrs['value'] = value
        attrs['track_value'] = track_value
        attrs['continuous_update'] = continuous_update
        self.attrs = attrs
class Select(ChangeTracker):
    base=JHTML.Select
    def __init__(self, var, options, value=None, **attrs):
        self._options = self.canonicalize_options(options)
        if value is None and len(self._options) > 0:
            value = self._options[0][1]
        super().__init__(var, value=value, **attrs)
    @classmethod
    def canonicalize_options(cls, options):
        ops = []
        for k in options:
            if isinstance(k, str):
                v = k
            else:
                try:
                    k, v = k
                except ValueError:
                    v = k
            ops.append((k,v))
        return tuple(ops)
    def _build_options_list(self):
        return [JHTML.Option(k, value=v) for k,v in self._options]
    def to_jhtml(self):
        field = self.base(*self._build_options_list(), **self.attrs)
        return field

class ListPicker(ChangeTracker):
    base = ListGroup
    def __init__(self, var, options, **attrs):
        super().__init__(var)

    def canonicalize_options(self):
        ...

#endregion

#region Layouts

class GridItem(Component):
    # __slots__ = ['item', "row", "col", 'span_rows', 'span_cols', 'alignment', 'justification', 'attrs']
    def __init__(self, item,
                 row=None, col=None,
                 row_span=None, col_span=None,
                 alignment=None, justification=None,
                 **attrs
                 ):
        super().__init__()
        self.item = item
        self.row = row
        self.col = col
        self.row_span = row_span
        self.col_span = col_span
        self.alignment = alignment
        self.justification = justification
        self.attrs = attrs
    @classmethod
    def get_grid_styles(cls,
                           row=None, row_span=None,
                           col=None, col_span=None,
                           alignment=None, justification=None
                           ):
        settings = {}
        if row is not None:
            settings['grid-row-start'] = row
            if row_span is not None:
                settings['grid-row-end'] = 'span ' + str(row + row_span)
        if col is not None:
            settings['grid-column-start'] = col
            if col_span is not None:
                settings['grid-column-end'] = 'span ' + str(col + col_span)
        if alignment is not None:
            settings['align-self'] = alignment
        if justification is not None:
            settings['justify-self'] = justification
        return settings
    @property
    def styles(self):
        return self.get_grid_styles(
            row=self.row, row_span=self.row_span,
            col=self.col, col_span=self.col_span,
            alignment=self.alignment, justification=self.justification,
        )
    def to_jhtml(self):
        attrs = self.attrs
        style = self.styles
        if 'style' in self.attrs:
            style = dict(attrs['style'], **self.styles)
            attrs = attrs.copy()
            del attrs['style']
        wat = JHTML.Div(
            self.item,
            style=style,
            **attrs
        )
        return wat

class Grid(Component):
    Item = GridItem
    def __init__(self, elements,
                 rows=None, cols=None,
                 alignment=None, justification=None,
                 row_spacing=None, col_spacing=None,
                 item_attrs=None,
                 row_height='1fr',
                 column_width='1fr',
                 style=None,
                 **attrs
                 ):
        super().__init__()
        if item_attrs is None:
            item_attrs = {}
        r, c, self.elements = self.setup_grid(elements, item_attrs)
        if rows is None:
            rows = r
        if cols is None:
            cols = c
        self.rows = rows
        self.cols = cols
        self.alignment = alignment
        self.justification = justification
        self.row_gaps = row_spacing
        self.col_gaps = col_spacing
        self.row_height = row_height
        self.col_width = column_width
        self._style = {} if style is None else style
        self.attrs = attrs

    @classmethod
    def setup_grid(cls, grid, attrs):
        elements = []
        nrows = 0
        ncols = 0
        for i, row in enumerate(grid):
            for j, el in enumerate(row):
                elem = cls.canonizalize_element(i+1, j+1, el, attrs)
                n = elem.row
                if elem.row_span is not None:
                    n += elem.row_span
                if n > nrows:
                    nrows = n
                m = elem.col
                if elem.col_span is not None:
                    m += elem.col_span
                if m > ncols:
                    ncols = m
                elements.append(elem)

        return nrows, ncols, elements

    @classmethod
    def canonizalize_element(cls, i, j, e, attrs):
        if not isinstance(e, GridItem):
            e = GridItem(e, row=i, col=j, **attrs)
        elif hasattr(e, 'items'):
            body = e['body']
            e = dict(e)
            del e['body']
            if 'row' not in e:
                e['row'] = i
            if 'col' not in e:
                e['col'] = j
            e = GridItem(body, **dict(attrs, **e))
        else:
            if e.row is None:
                e.row = i
            if e.col is None:
                e.col = j
        return e

    @classmethod
    def get_grid_styles(cls,
                        rows=None,  cols=None,
                        alignment=None, justification=None,
                        row_gap=None, col_gap=None,
                        row_height='1fr', col_width='1fr'
                        ):
        settings = {'display':'grid'}
        if rows is not None:
            settings['grid-template-rows'] = 'repeat({rows}, {height})'.format(rows=rows, height=row_height) if isinstance(rows, int) else rows
        if cols is not None:
            settings['grid-template-columns'] = 'repeat({cols}, {width})'.format(cols=cols, width=col_width) if isinstance(cols, int) else cols
        if alignment is not None:
            settings['align-items'] = alignment
        if justification is not None:
            settings['justify-items'] = justification
        if row_gap is not None:
            settings['row-gap'] = row_gap
        if row_gap is not None:
            settings['column-gap'] = col_gap
        return settings

    @property
    def styles(self):
        return dict(self._style, **self.get_grid_styles(
            rows=self.rows, cols=self.cols,
            alignment=self.alignment, justification=self.justification,
            row_gap=self.row_gaps, col_gap=self.col_gaps,
            row_height=self.row_height, col_width=self.col_width
        ))

    def to_jhtml(self):
        return JHTML.Div(
            *self.elements,
            style=self.styles,
            **self.attrs
        )

#endregion

# class CardManipulator(WidgetInterface):
#     def __init__(self, func, *controls, title=None, output_pane=None, **opts):
#         self.controls = [Manipulator.canonicalize_control(c) for c in controls]
#         vars = [c.var for c in self.controls]
#         self.output = FunctionDisplay(func, vars)
#         # really I want to do this by layout but this works for now...
#         self.toolbar_controls = [x for x in self.controls if not hasattr(x, 'layout_orientation') or x.layout_orientation != 'column']
#         self.column_controls = [x for x in self.controls if hasattr(x, 'layout_orientation') and x.layout_orientation == 'column']
#         self.title = title
#         self.output_pane = output_pane
#
#     def to_widget(self):
#         interface = JHTML.Bootstrap.Card(
#             JHTML.Bootstrap.CardHeader(
#                 "" if self.title is None else self.title
#                 # JHTML.SubsubHeading("A Pane with Output", JHTML.Small(" with a slider for fun", cls='text-muted')),
#             ),
#             JHTML.Bootstrap.CardBody(
#                 JHTML.Div(
#                     JHTML.Bootstrap.Row(
#                         *(JHTML.Bootstrap.Col(c.to_widget(), cls=["col-1", 'bg-light', 'border-end', 'p-0', 'm-0'])
#                           for c in self.column_controls),
#                         JHTML.Bootstrap.Col(
#                             JHTML.Div(
#                                 *[t.to_widget() for t in self.toolbar_controls],
#                                 cls=['bg-light', 'border-bottom', 'd-inline-block', 'w-100', "p-2"]
#                                 # , style=dict(min_height="80px")#, flex="1")
#                             ),
#                             self.output.to_widget(),
#                             style=dict(min_height='500px')
#                         ),
#                         cls=['g-0', "flex-grow-1"]
#                     ),
#                     style=dict(flex="1"),
#                     cls=["p-0", "d-flex", "flex-column", 'max-width-auto', 'w-100']
#                 ),
#                 cls=["p-0"]
#             ),
#             JHTML.Bootstrap.CardFooter("" if self.output_pane is None else self.output_pane)
#         )
#
#         return interface
#     def initialize(self):
#         self.output.update()

# class Form(Component):
#
#     def __init__(self, *form_specs, layout_function=None):
#         self.forms = form_specs
#         self.layout = layout_function
#
#     @classmethod
#     def create_control(self,
#                        name=None,
#                        field_type=None,
#                        default_value=None,
#                        control_type=None,
#                        label=True
#                        ):
#         ...

    # def