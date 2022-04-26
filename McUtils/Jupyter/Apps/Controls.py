
import abc, uuid, numpy as np
from ..JHTML import JHTML

from .Interfaces import Component, ListGroup, Dropdown
from .Variables import Var

__all__ = [
    "Control",
    "InputField",
    "StringField",
    "Checkbox",
    "Slider",
    "RadioButton",
    "TextArea",
    "Select",
    "VariableDisplay",
    # "DropdownMenu",
    "MenuSelect",
    "DropdownSelect"
]

__reload_hook__ = ['.Interfaces', '.Variables']

#region JHTML Controls
class Control(Component):
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

    control_types = {}
    @classmethod
    def construct(cls, var, control_type=None, **attrs):
        if control_type is None:
            control_type = cls.infer_control(**attrs)
        if isinstance(control_type, str):
            control_type = cls.control_types[control_type]
        return control_type
    @classmethod
    def infer_control(cls, field_type=None, default=None, **ignored):
        if field_type is None and default is not None:
            field_type = type(default)
        if field_type is not None:
            if issubclass(field_type, str):
                return cls.control_types['StringField']
            elif issubclass(field_type, (bool,)):
                return cls.control_types['Checkbox']
            elif issubclass(field_type, (int, np.integer, float, np.floating)):
                return cls.control_types['Slider']
            else:
                raise NotImplementedError("can't infer control type for 'field_type' {}".format(field_type))
        else:
            raise NotImplementedError("can't infer control type without 'field_type'")

class ValueWidget(Control):
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
        attrs['track_value'] = track_value
        attrs['continuous_update'] = continuous_update
        self.attrs = attrs
    def to_jhtml(self):
        self._attrs['value'] = self.var.value
        field = JHTML.Input(**self.attrs)
        return field
class StringField(InputField):
    def __init__(self, var, type='text', **attrs):
        super().__init__(var, type=type, **attrs)
Control.control_types['StringField'] = StringField
class Slider(InputField):
    def __init__(self, var, type='range', **attrs):
        super().__init__(var, type=type, **attrs)
Control.control_types['Slider'] = Slider
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
Control.control_types['Checkbox'] = Checkbox
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
class VariableDisplay(Control):
    def __init__(self, var, pane=None, autoclear=True, **attrs):
        super().__init__(var)
        if pane is None:
            pane = JHTML.OutputArea(autoclear=autoclear, **attrs)
        self.out = pane
    def get_value(self):
        return self.var.value
    def set_value(self):
        val = self.var.value
        if val is not None:
            if isinstance(val, str):
                if val == "":
                    self.out.clear()
                else:
                    self.out.print(val)
            else:
                self.out.show_output(val)
        else:
            self.out.clear()
    def update(self, e):
        self.set_value()
    def to_jhtml(self):
        self.set_value()
        return self.out

class MenuSelect(ValueWidget):
    menu_type = ListGroup
    def __init__(self, var, options, menu_type=None, **attrs):
        super().__init__(var)
        self._value_map, self.ops = self.canonicalize_options(options)
        if menu_type is not None:
            self.menu_type = menu_type
        self._obj = self.menu_type(self.ops, **attrs) #type: MenuComponent
        self._active_item = None
    def get_value(self):
        if self._active_item is not None:
            return self._value_map[self._active_item]
    def set_value(self):
        self.set_active(self.var.value)
    def update(self, e):
        self.set_active(self.value)
    def set_active(self, v):
        for k, test in self._value_map.items():
            if test == v:
                break
        else:
            k = None
        if k is not None:
            self.set_active_key(k)
    def set_active_key(self, k):
        if len(self._obj._item_map) > 0:
            if (
                    self._active_item is not None
                    and k != self._active_item
            ):
                cur = self._obj._item_map[self._active_item]
                if cur is not None:
                    cur.remove_class('active')
            self._active_item = k
            new = self._obj._item_map[k]
            if new is not None:
                new.add_class('active')
    def onclick(self, e, i, v):
        self.var.set_value(v, caller=self)
        # with self.logger_pane:
        self.set_active(v)

    def canonicalize_options(self, options):
        ops = []
        val_dict = {}
        for k in options:
            if not isinstance(k, dict):
                if isinstance(k, str):
                    v = k
                else:
                    try:
                        k, v = k
                    except (TypeError, ValueError):
                        v = k
                k = {
                    'body':k,
                    'value':v
                }
            if 'id' not in k:
                uid = str(uuid.uuid1()).replace("-", "")
                k['id'] = uid
            val_dict[k['id']] = k['value']
            k['event_handlers'] = {'click':lambda *e,i=k['id'],v=k['value']:self.onclick(e, i, v)}
            ops.append(k)
        return val_dict, tuple(ops)
    def to_jhtml(self):
        widg = self._obj.to_jhtml()
        init_key = next(iter(self._value_map.keys()))
        self.set_active(init_key)
        return widg
class DropdownSelect(ValueWidget):
    menu_type = Dropdown
    def __init__(self, var, options, name=None, menu_type=None, **attrs):
        super().__init__(var)
        if menu_type is not None:
            self.menu_type = menu_type
        self.selector = MenuSelect(var, options, menu_type=self.menu_type.List)
        if name is None:
            name = self.var.name
        self.name = name
        self.attrs = attrs

    def get_value(self):
        return self.selector.get_value()
    def set_value(self):
        self.selector.set_value()
    def update(self, e):
        self.selector.update(e)
    def to_jhtml(self):
        return self.menu_type(
            self.name,
            self.selector.to_widget(),
            **self.attrs
        ).to_jhtml()
