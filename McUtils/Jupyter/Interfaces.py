
import abc, numpy as np, weakref, uuid
from .JHMTL import JHTML
from .WidgetTools import JupyterAPIs

__all__ = [
    "Var",
    "Control",
    "Manipulator",
    "Sidebar",
    "SidebarSetter"
]

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

class OutputWidget:
    def __init__(self, var, **ignored):
        self.var = var
        self.output = JupyterAPIs.get_display_api().Output()
        self.display(self.var.value)
    def on_change(self):
        self.display(self.var.value)
    def print(self, *args, **kwargs):
        with self:
            print(*args, **kwargs)
    def display(self, *args):
        with self:
            JupyterAPIs.get_display_api().display_api.display(*args)
    def __enter__(self):
        self.output.clear_output()
        return self.output.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.output.__exit__(exc_type, exc_val, exc_tb)
    def to_widget(self):
        return self.output

class Var:
    _var_cache = weakref.WeakValueDictionary()
    def __init__(self, name, value=None, callbacks=()):
        self._name = name
        self._value = value
        self.callbacks = weakref.WeakSet(callbacks)
        self._watchers = weakref.WeakSet()
    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._name,
            self._value
        )
    @classmethod
    def create_var(cls, var):
        if isinstance(var, Var):
            return var
        else:
            if var not in cls._var_cache:
                this_var = Var(var)
                cls._var_cache[var] = this_var # hold a reference...
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
        old = self._value
        self._value = v
        for c in self.callbacks:
            c({'var': self, 'old': old, 'new': self._value})
        for w in self._watchers:
            if w is not caller:
                w.value = self._value
    def link(self, widget):
        self.set_value(widget.value, caller=widget)
        widget.observe(lambda d: self.set_value(widget.value, caller=widget))
        self._watchers.add(widget)

class Control:
    def __init__(self, var, control_type=None, widget=None, **settings):
        self.var = Var.create_var(var)
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
                control_type = OutputWidget
        return control_type(**settings)

    def to_widget(self):
        self.var.link(self.widget)
        return self.widget

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

class Manipulator:
    def __init__(self, func, *controls):
        self.controls = [self.canonicalize_control(c) for c in controls]
        vars = [c.var for c in self.controls]
        self.output = FunctionDisplay(func, vars)
    @classmethod
    def canonicalize_control(cls, settings):
        if isinstance(settings, Control):
            return settings
        else:
            return Control(settings[0], **settings[1])
    def to_widget(self):
        widgets = JupyterAPIs.get_widgets_api()
        elems = [c.to_widget() for c in self.controls] + [self.output.to_widget()]
        # print(elems)
        return widgets.VBox(elems)
    def _ipython_display_(self):
        JupyterAPIs.get_display_api().display(self.to_widget())
        # with self.output.output:
        #     print("????")
        self.output.update()
    def display(self):
        self._ipython_display_()


class InterfaceElement(metaclass=abc.ABCMeta):
    """
    Provides an abstract base class for an interface element
    to allow for the easy construction of interesting interfaces
    """

    @abc.abstractmethod
    def to_jhtml(self):
        ...

    @staticmethod
    def manage_cls(cls):
        if isinstance(cls, str):
            cls = cls.split()
        elif cls is None:
            cls = []
        return list(cls)
class Sidebar(InterfaceElement):
    def __init__(self, *links, cls=None, item_attrs=None, **attrs):
        self.links = links
        self.cls = self.manage_cls(cls)
        self.attrs = attrs
        self.item_attrs = {} if item_attrs is None else item_attrs
    @classmethod
    def format_link(self, body=(), href="#", cls=None, **attrs):
        if isinstance(cls, str):
            cls = cls.split()
        return JHTML.Anchor(
            body,
            href=href,
            cls=['list-group-item', 'list-group-item-action'] + self.manage_cls(cls),
            **attrs
        )
    def to_jhtml(self):
        return JHTML.Div(
            JHTML.Div(
                *[self.format_link(**dict(self.item_attrs, **link)) for link in self.links],
                cls=['list-group', 'list-group-flush'] + self.cls
            ),
            cls=['d-flex', 'flex-column', 'align-items-stretch', 'flex-shrink-0'],
            **self.attrs
        )
class WidgetControl:
    def __init__(self, var):
        self.var = Var.create_var(var)
        self._widget_cache = None
        self._parents = weakref.WeakSet()
    @property
    def value(self):
        return self.var.value
    @value.setter
    def value(self, v):
        self.var.set_value(v, caller=self)
    def to_widget(self, parent=None):
        if parent is not None:
            self._parents.add(parent)
        if self._widget_cache is None:
            self._widget_cache = self.to_jhtml()
        return self._widget_cache
    @abc.abstractmethod
    def to_jhtml(self):
        ...

class SidebarSetter(WidgetControl):
    def __init__(self, var, options, item_attrs=None, logger_pane=None, **attrs):
        super().__init__(var)
        self.options = options
        self.item_attrs = {} if item_attrs is None else item_attrs
        self.attrs = attrs
        self._item_map = {}
        self._active_item = None
        self.logger_pane = logger_pane
        val = self.options[0]['value']
        self.var.value = val
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
                    self._widget_cache is not None
                    and self._active_item is not None
                    and k!=self._active_item
            ):
                tree = self._widget_cache
                cur = tree.find_by_id(self._active_item)
                if cur is not None:
                    cur.remove_class('active', copy=False)
                self._active_item = k
                new = tree.find_by_id(self._active_item)
                if new is not None:
                    new.add_class('active', copy=False)
    def onclick(self, e, i, v):
        self.var.set_value(v, caller=self)
        if self.logger_pane is not None:
            with self.logger_pane:
                self.set_active(v)
        # e, html, widg = e
        # widg.dom.tree[0].add_class('active', copy=False)
        # cur = widg.dom.get_parent(2).find_by_id(self._active_item)
        # if cur is not None:
        #     cur.remove_class('active', copy=False)
        # self._active_item = i
        # self.value = v
    def create_link_dict(self, o, which=-1):
        id = str(uuid.uuid1()).replace("-", "")
        o = o.copy()
        label = o['label'] if 'label' in o else o['value']
        if 'label' in o:
            del o['label']
        value = o['value']
        self._item_map[id] = value
        del o['value']
        if which == 0:
            self._active_item = id
            if 'cls' in o:
                cls = o['cls']
            elif 'cls' in self.item_attrs:
                cls = self.item_attrs['cls']
            else:
                cls = []
            if isinstance(cls, str):
                cls = cls.split()
            o['cls'] = list(cls) + ['active']
        o['id'] = id
        return dict(
            body=label,
            event_handlers=dict(
                click=lambda *e, i=id,v=value: self.onclick(e, i, v)
            ),
            **o
        )
    def to_jhtml(self):
        self.var.callbacks.add(self.update)
        return Sidebar(
            *[
                self.create_link_dict(o, which=n)
                for n,o in enumerate(self.options)
            ],
            item_attrs=self.item_attrs,
            **self.attrs
        ).to_jhtml()