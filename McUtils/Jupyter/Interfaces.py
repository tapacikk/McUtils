
import abc, numpy as np, weakref, uuid
from .JHMTL import JHTML
from .WidgetTools import JupyterAPIs, DefaultOutputArea

__all__ = [
    "Var",
    "InterfaceVars",
    "VariableSynchronizer",
    "Control",
    "Manipulator",
    "Sidebar",
    "SidebarSetter",
    "CardManipulator",
    "InputField",
    "StringField",
    "Checkbox",
    "Slider",
    "RadioButton",
    "TextArea",
    "Select"
]
__reload_hook__ = [".JHTML", ".WidgetTools"]

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


class SettingChecker:
    int_types = (int, np.integer)
    numerics_types = (int, np.integer, float, np.floating)
    control_type = None
    @classmethod
    def check(self, **props):
        return False
    checkers = []
class CheckboxChecker(SettingChecker):
    control_type = "Checkbox"
    @classmethod
    def check(self, value=None, **rest):
        return isinstance(value, bool)
SettingChecker.checkers.append(CheckboxChecker)
class DropdownChecker(SettingChecker):
    control_type = "Dropdown"
    @classmethod
    def check(self, options=None, **rest):
        return options is not None
SettingChecker.checkers.append(DropdownChecker)
class TextChecker(SettingChecker):
    control_type = "Text"
    @classmethod
    def check(self, value=None, **rest):
        return isinstance(value, str)
SettingChecker.checkers.append(TextChecker)
class IntSliderChecker(SettingChecker):
    control_type = "IntSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return isinstance(value, self.int_types) and isinstance(min, self.int_types) and isinstance(max, self.int_types)
SettingChecker.checkers.append(IntSliderChecker)
class FloatSliderChecker(SettingChecker):
    control_type = "FloatSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return (
                isinstance(value, self.numerics_types)
                and isinstance(min, self.numerics_types)
                and isinstance(max, self.numerics_types)
        )
SettingChecker.checkers.append(FloatSliderChecker)
class IntRangeChecker(SettingChecker):
    control_type = "IntRangeSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return (
                len(value) == 2
                and isinstance(value[0], self.int_types)
                and isinstance(value[1], self.int_types)
                and isinstance(min, self.int_types) and isinstance(max, self.int_types)
        )
SettingChecker.checkers.append(IntRangeChecker)
class FloatRangeChecker(SettingChecker):
    control_type = "FloatRangeSlider"
    @classmethod
    def check(self, value=None, min=None, max=None, **rest):
        return (
                len(value) == 2
                and isinstance(value[0], self.numerics_types)
                and isinstance(value[1], self.numerics_types)
                and isinstance(min, self.numerics_types) and isinstance(max, self.numerics_types)
        )
SettingChecker.checkers.append(FloatRangeChecker)

class InterfaceVars:
    _cache_stack = []
    def __init__(self):
        self._var_set = set()
        self.var_list = []
    @classmethod
    def active_vars(cls):
        if len(cls._cache_stack) > 0:
            return cls._cache_stack[-1]
        else:
            return None
    def add(self, var):
        if var not in self._var_set:
            self.var_list.append(var)
            self._var_set.add(var)
    def __enter__(self):
        self._cache_stack.append(self)
        return self.var_list
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache_stack.pop()
class VariableSynchronizer:
    def __init__(self, name, value=None, callbacks=(), output_pane=None):
        self._name = name
        self._value = value
        self.callbacks = weakref.WeakSet(callbacks)
        self.output_pane = DefaultOutputArea.get_default() if output_pane is None else output_pane
        self._watchers = weakref.WeakSet()
    def __repr__(self):
        return "{}({}, {!r})".format(
            type(self).__name__,
            self._name,
            self._value
        )
    _var_cache = weakref.WeakValueDictionary()
    @classmethod
    def create_var(cls, var):
        var_cache = InterfaceVars.active_vars()
        if isinstance(var, VariableSynchronizer):
            if var_cache is not None:
                var_cache.add(var)
            return var
        else:
            if var not in cls._var_cache:
                this_var = VariableSynchronizer(var)
                cls._var_cache[var] = this_var # hold a reference...
            if var_cache is not None:
                var_cache.add(cls._var_cache[var])
            return cls._var_cache[var]
    @property
    def name(self):
        return self._name
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, v):
        self.set_value(v)
    def set_value(self, v, caller=None):
        with self.output_pane:
            old = self._value
            try:
                check = old != v
            except TypeError:
                check = old is not v
            if check:
                self._value = v
                for c in self.callbacks:
                    c({'var': self, 'old': old, 'new': self._value})
                for w in self._watchers:
                    if w is not caller:
                        w.value = self._value
    def link(self, widget):
        self.set_value(widget.value, caller=widget)
        widget.observe(lambda d: self.set_value(widget.value, caller=widget), names=['value'])
        self._watchers.add(widget)
def Var(name):
    return VariableSynchronizer.create_var(name)
class Control:
    def __init__(self, var, control_type=None, widget=None, **settings):
        self.var = VariableSynchronizer.create_var(var)
        self.settings = settings
        if widget is None:
            widget = self._build_widget(control_type, settings)
        self.widget = widget
    @classmethod
    def _build_widget(cls, control_type, settings):
        if control_type is None:
            for checker in SettingChecker.checkers:
                if checker.check(**settings):
                    control_type = getattr(JupyterAPIs.get_widgets_api(), checker.control_type)
                    break
            else:
                control_type = JHTML.OutputArea
        return control_type(**settings)

    def to_widget(self):
        self.var.link(self.widget)
        return self.widget

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
        # print(elems)
        return widgets.VBox(elems)
    def initialize(self):
        self.output.update()

class JHTMLInterface(WidgetInterface):
    """
    Provides an abstract base class for an interface element
    to allow for the easy construction of interesting interfaces
    """
    def __init__(self):
        self._parents = weakref.WeakSet()

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

# class TabBar(JHTMLInterface):
#     ...
#
#
#
#     """
#     <ul class="nav nav-tabs">
#   <li class="nav-item">
#     <a class="nav-link active" aria-current="page" href="#">Active</a>
#   </li>
#   <li class="nav-item dropdown">
#     <a class="nav-link dropdown-toggle" data-bs-toggle="dropdown" href="#" role="button" aria-expanded="false">Dropdown</a>
#     <ul class="dropdown-menu">
#       <li><a class="dropdown-item" href="#">Action</a></li>
#       <li><a class="dropdown-item" href="#">Another action</a></li>
#       <li><a class="dropdown-item" href="#">Something else here</a></li>
#       <li><hr class="dropdown-divider"></li>
#       <li><a class="dropdown-item" href="#">Separated link</a></li>
#     </ul>
#   </li>
#   <li class="nav-item">
#     <a class="nav-link" href="#">Link</a>
#   </li>
#   <li class="nav-item">
#     <a class="nav-link disabled">Disabled</a>
#   </li>
# </ul>
#     """
class Menulike(JHTMLInterface):
    def __init__(self, *items, item_attrs=None):
        super().__init__()
        self.items = items
        self._item_map = {}
        self.item_attrs = {} if item_attrs is None else item_attrs
    @abc.abstractmethod
    def create_item(self, body=None, id=None, **extra):
        ...
    @abc.abstractmethod
    def wrap_items(self, items):
        ...
    def to_jhtml(self, parent=None):
        items = [self.create_item(**dict(self.item_attrs, **item)) for item in self.items]
        for i in items:
            if hasattr(i, 'id'):
                self._item_map[i.id] = i
            elif 'id' in i.attrs:
                self._item_map[i.attrs['id']] = i
        return self.wrap_items(items)

class Sidebar(JHTMLInterface):
    item_classes = ['list-group-item', 'list-group-item-action']
    wrapper_classes = ['list-group', 'list-group-flush']
    base_classes = ['d-flex', 'flex-column', 'align-items-stretch', 'flex-shrink-0']
    def __init__(self, *links, cls=None,
                 item_attrs=None,
                 item_classes=None,
                 wrapper_classes=None,
                 base_classes=None,
                 **attrs):
        super().__init__()
        self.links = links
        self.cls = JHTML.manage_cls(cls)
        self.attrs = attrs
        if item_classes is None:
            item_classes = self.item_classes
        self.item_classes = item_classes
        if wrapper_classes is None:
            wrapper_classes = self.wrapper_classes
        self.wrapper_classes = wrapper_classes
        if base_classes is None:
            base_classes = self.base_classes
        self.base_classes = base_classes
        self.item_attrs = {} if item_attrs is None else item_attrs
    @classmethod
    def format_link(self, body=(), href="#", cls=None, **attrs):
        return JHTML.Anchor(
            body,
            href=href,
            cls=self.item_classes + JHTML.manage_cls(cls),
            **attrs
        )
    def to_jhtml(self):
        return JHTML.Div(
            JHTML.Div(
                *[self.format_link(**dict(self.item_attrs, **link)) for link in self.links],
                cls=self.wrapper_classes + self.cls
            ),
            cls=self.base_classes,
            **self.attrs
        )
class DropdownMenu(JHTMLInterface):
    def __init__(self, actions):
        self.actions = actions
        super().__init__()
    def create_item(self, body=None, href="#", id=None, action=None):
        attrs = {'href':href}
        if action is not None:
            attrs['event_handlers'] = {'click':action}
        if body is not None:
            body = [body]
        return JHTML.ListItem(
            JHTML.Anchor(*body, **attrs)
        )


class JHTMLControl(JHTMLInterface):
    layout_orientation = 'row'
    def __init__(self, var):
        self.var = VariableSynchronizer.create_var(var)
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
class ActiveSetter(JHTMLControl):
    def __init__(self, var, options, item_attrs=None, debug_pane=None, **attrs):
        super().__init__(var)
        self.options = options
        self.item_attrs = {} if item_attrs is None else item_attrs
        self.attrs = attrs
        self._element_map = {}
        self._item_map = {}
        self._active_item = None
        self.logger_pane = DefaultOutputArea() if debug_pane is None else debug_pane
        val = self.options[0]['value']
        self.var.value = val
    def get_value(self):
        return self._item_map[self._active_item]
    def set_value(self):
        self.set_active(self.var.value)
    def update(self, e):
        self.set_active(self.value)
    def set_active(self, v):
        for k, test in self._item_map.items():
            if test == v:
                break
        else:
            k = None

        if k is not None:
            if (
                    self._active_item is not None
                    and k != self._active_item
                    and len(self._element_map) > 0
            ):
                cur = self._element_map[self._active_item]
                if cur is not None:
                    cur.remove_class('active')
                self._active_item = k
                new = self._element_map[k]
                if new is not None:
                    new.add_class('active')
    def onclick(self, e, i, v):
        self.var.set_value(v, caller=self)
        with self.logger_pane:
            self.set_active(v)

    @abc.abstractmethod
    def create_items(self):
        ...

    # def create_link_dict(self, o, which=-1):
    #     uid = str(uuid.uuid1()).replace("-", "")
    #     o = o.copy()
    #     label = o['label'] if 'label' in o else o['value']
    #     if 'label' in o:
    #         del o['label']
    #     value = o['value']
    #     self._item_map[uid] = value
    #     del o['value']
    #     if which == 0:
    #         self._active_item = uid
    #         if 'cls' in o:
    #             cls = o['cls']
    #         elif 'cls' in self.item_attrs:
    #             cls = self.item_attrs['cls']
    #         else:
    #             cls = []
    #         if isinstance(cls, str):
    #             cls = cls.split()
    #         o['cls'] = list(cls) + ['active']
    #     o['id'] = uid
    #     return dict(
    #         body=label,
    #         event_handlers=dict(
    #             click=lambda *e, i=uid, v=value: self.onclick(e, i, v)
    #         ),
    #         **o
    #     )

    def to_jhtml(self):
        bar = Sidebar(
            *[
                self.create_link_dict(o, which=n)
                for n, o in enumerate(self.options)
            ],
            item_attrs=self.item_attrs,
            **self.attrs
        ).to_jhtml()

        col_wrapper = bar.get_child(0, wrapper=True)
        for i in range(len(col_wrapper.children)):
            wrapper = col_wrapper.get_child(i, wrapper=True)
            self._element_map[wrapper['id']] = wrapper

        return bar
class SidebarSetter(JHTMLControl):
    layout_orientation = 'column'
    def __init__(self, var, options, item_attrs=None, debug_pane=None, **attrs):
        super().__init__(var)
        self.options = options
        self.item_attrs = {} if item_attrs is None else item_attrs
        self.attrs = attrs
        self._element_map = {}
        self._item_map = {}
        self._active_item = None
        self.logger_pane = DefaultOutputArea() if debug_pane is None else debug_pane
        val = self.options[0]['value']
        self.var.value = val
    def get_value(self):
        return self._item_map[self._active_item]
    def set_value(self):
        self.set_active(self.var.value)
    def update(self, e):
        self.set_active(self.value)
    def set_active(self, v):
        for k,test in self._item_map.items():
            if test == v:
                break
        else:
            k = None

        if k is not None:
            if (
                self._active_item is not None
                and k != self._active_item
                and len(self._element_map) > 0
            ):
                cur = self._element_map[self._active_item]
                if cur is not None:
                    cur.remove_class('active')
                self._active_item = k
                new = self._element_map[k]
                if new is not None:
                    new.add_class('active')
    def onclick(self, e, i, v):
        self.var.set_value(v, caller=self)
        with self.logger_pane:
            self.set_active(v)
    def create_link_dict(self, o, which=-1):
        uid = str(uuid.uuid1()).replace("-", "")
        o = o.copy()
        label = o['label'] if 'label' in o else o['value']
        if 'label' in o:
            del o['label']
        value = o['value']
        self._item_map[uid] = value
        del o['value']
        if which == 0:
            self._active_item = uid
            if 'cls' in o:
                cls = o['cls']
            elif 'cls' in self.item_attrs:
                cls = self.item_attrs['cls']
            else:
                cls = []
            if isinstance(cls, str):
                cls = cls.split()
            o['cls'] = list(cls) + ['active']
        o['id'] = uid
        return dict(
            body=label,
            event_handlers=dict(
                click=lambda *e, i=id,v=value: self.onclick(e, i, v)
            ),
            **o
        )
    def to_jhtml(self):
        bar = Sidebar(
            *[
                self.create_link_dict(o, which=n)
                for n,o in enumerate(self.options)
            ],
            item_attrs=self.item_attrs,
            **self.attrs
        ).to_jhtml()

        col_wrapper = bar.get_child(0, wrapper=True)
        for i in range(len(col_wrapper.children)):
            wrapper = col_wrapper.get_child(i, wrapper=True)
            self._element_map[wrapper['id']] = wrapper

        return bar
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
    base='Select'
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

class CardManipulator(WidgetInterface):
    def __init__(self, func, *controls, title=None, output_pane=None, **opts):
        self.controls = [Manipulator.canonicalize_control(c) for c in controls]
        vars = [c.var for c in self.controls]
        self.output = FunctionDisplay(func, vars)
        # really I want to do this by layout but this works for now...
        self.toolbar_controls = [x for x in self.controls if not hasattr(x, 'layout_orientation') or x.layout_orientation != 'column']
        self.column_controls = [x for x in self.controls if hasattr(x, 'layout_orientation') and x.layout_orientation == 'column']
        self.title = title
        self.output_pane = output_pane

    def to_widget(self):
        interface = JHTML.Bootstrap.Card(
            JHTML.Bootstrap.CardHeader(
                "" if self.title is None else self.title
                # JHTML.SubsubHeading("A Pane with Output", JHTML.Small(" with a slider for fun", cls='text-muted')),
            ),
            JHTML.Bootstrap.CardBody(
                JHTML.Div(
                    JHTML.Bootstrap.Row(
                        *(JHTML.Bootstrap.Col(c.to_widget(), cls=["col-1", 'bg-light', 'border-end', 'p-0', 'm-0'])
                          for c in self.column_controls),
                        JHTML.Bootstrap.Col(
                            JHTML.Div(
                                *[t.to_widget() for t in self.toolbar_controls],
                                cls=['bg-light', 'border-bottom', 'd-inline-block', 'w-100', "p-2"]
                                # , style=dict(min_height="80px")#, flex="1")
                            ),
                            self.output.to_widget(),
                            style=dict(min_height='500px')
                        ),
                        cls=['g-0', "flex-grow-1"]
                    ),
                    style=dict(flex="1"),
                    cls=["p-0", "d-flex", "flex-column", 'max-width-auto', 'w-100']
                ),
                cls=["p-0"]
            ),
            JHTML.Bootstrap.CardFooter("" if self.output_pane is None else self.output_pane)
        )

        return interface
    def initialize(self):
        self.output.update()

class Form(WidgetInterface):

    def __init__(self, *form_specs, layout_function=None):
        self.forms = form_specs
        self.layout = layout_function

    @classmethod
    def create_control(self,
                       name=None,
                       field_type=None,
                       default_value=None,
                       control_type=None,
                       label=True
                       ):
        ...

    # def


# class GridItem(WidgetInterface):
#     __slots__ = ['item', "row", "col", 'span_rows', 'span_cols']
#     def __init__(self, item, row, col, span_rows=1, span_cols=1):
#         self.item = item
#         self.row = row
#         self.col = col
#         self.span_rows = span_rows
#         self.span_cols = span_cols
#     def to_jhtml(self):
#         return JHTML.Div(
#             self.item,
#             cls=
#         )
#
# class Grid(WidgetInterface):
#     def __init__(self, elements):
#         self.elements = elements
#
#     def compute_spans(self, i, j, span):
#         ...
#
#     def to_jhtml(self):
#         JHTML.Div(
#
#             cls='grid'
#         )

# class AppExplorer(WidgetInterface):
#
#     def __init__(self, interfaces):
#         self.interfaces = interfaces
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