
import abc, weakref, uuid
from ..JHTML import JHTML
from ..JHTML.WidgetTools import JupyterAPIs, frozendict

__all__ = [
    "WidgetInterface",
    "Component",
    "Container",
    "ListGroup",
    "Navbar",
    "Dropdown",
    "DropdownList",
    "Tabs",
    "TabPane",
    "TabList",
    "Layout",
    "Grid",
    "Flex"
]
__reload_hook__ = ["..JHTML", "..WidgetTools"]

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
    def __init__(self, items, wrapper=None, wrapper_classes=None, cls=None, **attrs):
        super().__init__(**attrs)
        if wrapper is not None:
            self.wrapper = wrapper
        if wrapper_classes is not None:
            self.wrapper_classes = wrapper_classes
        self.items = items
        self.wrapper_classes = self.wrapper_classes + JHTML.manage_cls(cls)
    def wrap_items(self, items):
        return self.wrapper(*items, cls=self.wrapper_classes, **self.attrs)
    def to_jhtml(self, parent=None):
        return self.wrap_items(self.items)
class Container(WrapperComponent):
    subwrappers = None
    subwrapper_classes = None
    item = JHTML.Span
    item_classes = []
    def __init__(self, items,
                 wrapper=None, wrapper_classes=None,
                 subwrappers=None, subwrapper_classes=None,
                 item=None, item_classes=None, item_attrs=None,
                 cls=None,
                 **attrs):

        self._items = None
        if subwrappers is None:
            subwrappers = self.subwrappers
        if subwrapper_classes is None:
            subwrapper_classes = self.subwrapper_classes
        if subwrappers is not None:
            if wrapper is None:
                wrapper = self.wrapper
            if wrapper_classes is not None:
                self.wrapper_classes = wrapper_classes
            if subwrapper_classes is None:
                subwrapper_classes = [None]*len(subwrappers)
            if len(subwrappers) != len(subwrapper_classes):
                raise ValueError("mismatch between number of subwrappers and wrapper classes")

            wrapper = JHTML.Compound(
                JHTML.Styled(wrapper, cls=self.wrapper_classes + JHTML.manage_cls(cls), **attrs),
                *(
                    JHTML.Styled(sw, cls=JHTML.manage_cls(sc))
                    for sw, sc in zip(subwrappers, subwrapper_classes)
                  )
            )
            self.wrapper_classes = JHTML.manage_cls(subwrapper_classes[-1])
            super().__init__(
                None,
                wrapper=wrapper
            )
        else:
            super().__init__(None, cls=cls, wrapper=wrapper, wrapper_classes=wrapper_classes, **attrs)

        self._items = items
        if item is not None:
            self.item = item
        if item_classes is not None:
            self.item_classes = item_classes
        if item_attrs is None:
            item_attrs = {}
        if 'cls' in item_attrs:
            item_attrs = item_attrs.copy()
            self.item_classes = self.item_classes + JHTML.manage_cls(item_attrs['cls'])
            del item_attrs['cls']
        self.item_attrs = item_attrs

    @property
    def items(self):
        return [self.create_item(i) for i in self._items]
    @items.setter
    def items(self, items):
        if self._items is not None:
            raise ValueError("can't set items")
    def _create_dict_item(self, body=None, cls=None, **extra):
        return self.item(body, cls=self.item_classes + JHTML.manage_cls(cls), **dict(self.item_attrs, **extra))
    def _create_base_item(self, body):
        return self.item(body, cls=self.item_classes, **self.item_attrs)
    def create_item(self, i, **kw):
        if isinstance(i, dict):
            if len(kw) > 0:
                return self._create_dict_item(**dict(i, **kw))
            else:
                return self._create_dict_item(**i)
        else:
            if len(kw) > 0:
                return self._create_dict_item(body=i, **kw)
            else:
               return self._create_base_item(i)

#region Menus

class MenuComponent(Container):
    def __init__(self, items, item_attrs=None, cls=None, **attrs):
        super().__init__(items, item_attrs=item_attrs, cls=cls, **attrs)
        self._item_map = {}
    def create_item(self, item, **kw):
        item = super().create_item(item, **kw)
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
    def create_item(self, item, **kw):
        item = super().create_item(item, **kw)
        if 'action' in item:
            item = item.copy()
            item['event_handlers'] = {'click':item['action']}
            del item['action']
        return item
class Dropdown(MenuComponent):
    wrapper_classes = ['dropdown']
    toggle_classes = ['dropdown-toggle']
    toggle = JHTML.Styled(JHTML.Bootstrap.Button, data_bs_toggle='dropdown')
    List = DropdownList
    def __init__(self, header, actions, toggle_attrs=None, **attrs):
        self.header = header
        self.toggle_attrs = {} if toggle_attrs is None else toggle_attrs
        if 'cls' in self.toggle_attrs:
            self.toggle_attrs = self.toggle_attrs.copy()
            self.toggle_classes = self.toggle_classes + JHTML.manage_cls(self.toggle_attrs['cls'])
            del self.toggle_attrs['cls']
        self.dlist = DropdownList(actions, **attrs) if not hasattr(actions, 'to_widget') else actions
        super().__init__([], **attrs)
    def wrap_items(self, items):
        items = [
            self.toggle(self.header, cls=self.toggle_classes, **self.toggle_attrs),
            self.dlist
        ]
        return super().wrap_items(items)

class Navbar(MenuComponent):
    wrapper = JHTML.Nav
    wrapper_classes = ['navbar', 'navbar-expand']
    subwrappers = [JHTML.Div, JHTML.Div]
    subwrapper_classes = [['container-fluid'], ['navbar-nav']]
    item = JHTML.Anchor
    item_classes = ['nav-link']

class TabList(MenuComponent):
    wrapper = JHTML.Ul
    wrapper_classes = ['nav', 'nav-tabs']
    item = JHTML.Compound(
        JHTML.Styled(JHTML.Li, cls='nav-item'),
        JHTML.Button
    )
    item_classes = ['nav-link']
    def __init__(self, *args, base_name=None, role="tablist", **kwargs):
        if base_name is None:
            base_name = 'tabs-' + str(uuid.uuid1()).replace("-", "")[:5]
        self.base_name = base_name
        self._active = None
        super().__init__(*args, role=role, **kwargs)
    def create_item(self, item, cls=None, **kw):
        item, _ = item
        item_id = self.base_name+"-"+item
        if self._active is None:
            cls = JHTML.manage_cls(cls) + ['active']
            self._active = item_id
        return super().create_item(item, id=item_id+'-tab', cls=cls, data_bs_target='#'+item_id, data_bs_toggle='tab', **kw)
class TabPane(MenuComponent):
    wrapper_classes = ['tab-content']
    item = JHTML.Div
    item_classes = ['tab-pane']
    def __init__(self, *args, base_name=None, **kwargs):
        if base_name is None:
            base_name = 'tabs-' + str(uuid.uuid1()).replace("-", "")[:5]
        self.base_name = base_name
        self._active = None
        super().__init__(*args, **kwargs)
    def create_item(self, item, cls=None, **kw):
        key, item = item
        item_id = self.base_name+"-"+key
        if self._active is None:
            cls = JHTML.manage_cls(cls) + ['active']
            self._active = item_id
        item = super().create_item(item, id=item_id, role='tabpanel', cls=cls, **kw)
        return item
class Tabs(MenuComponent):
    wrapper_classes = []
    # toggle_classes = ['dropdown-toggle']
    # toggle = JHTML.Styled(JHTML.Bootstrap.Button, data_bs_toggle='dropdown')
    def __init__(self, tabs, base_name=None, **attrs):
        if base_name is None:
            base_name = 'tabs-' + str(uuid.uuid1()).replace("-", "")[:5]
        if isinstance(tabs, dict):
            tabs = tabs.items()
        self.tab_list = TabList(tabs, id=base_name, base_name=base_name)
        self.panes = TabPane(tabs, id=base_name+'Content', base_name=base_name)
        super().__init__([], **attrs)
    def wrap_items(self, items):
        items = [
            self.tab_list,
            self.panes
        ]
        return super().wrap_items(items)

#endregion

#region Layouts

class LayoutItem(Component):
    wrapper = JHTML.Div
    properties = []
    def __init__(self, item, **attrs):
        super().__init__()
        self.item = item
        self.attrs = attrs
    @abc.abstractmethod
    def get_layout_styles(self, **kwargs):
        raise NotImplementedError("LayoutItem is an abstract class")
    def to_jhtml(self):
        attrs = self.attrs
        style = self.get_layout_styles()
        if 'style' in self.attrs:
            style = dict(attrs['style'], **style)
            attrs = attrs.copy()
            del attrs['style']
        wat = self.wrapper(
            self.item,
            style=style,
            **attrs
        )
        return wat
class Layout(Component):
    wrapper = JHTML.Div
    Item = LayoutItem
    def __init__(self, elements, wrapper=None, item_attrs=None, style=None, **attrs):
        super().__init__()
        if item_attrs is None:
            item_attrs = {}
        self.layout_settings, self.elements = self.setup_layout(elements, item_attrs)
        self._style = {} if style is None else style
        self.attrs = attrs
        if wrapper is None:
            wrapper = self.wrapper
        self.wrapper = wrapper
    def wrap_item(self, e, attrs):
        return self.Item(e, **attrs)
    def setup_layout(self, elements, item_attrs):
        return None, [self.wrap_item(e, item_attrs) for e in elements]
    @abc.abstractmethod
    def get_layout_styles(self, **kwargs):
        raise NotImplementedError("Layout is an abstract class")
    @property
    def styles(self):
        return dict(self._style, **self.get_layout_styles())
    def to_jhtml(self):
        return self.wrapper(
            *self.elements,
            style=self.styles,
            **self.attrs
        )

class GridItem(LayoutItem):
    def __init__(self, item,
                 row=None, col=None,
                 row_span=None, col_span=None,
                 alignment=None, justification=None,
                 **attrs
                 ):
        super().__init__(item, **attrs)
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
    def get_layout_styles(self):
        return self.get_grid_styles(
            row=self.row, row_span=self.row_span,
            col=self.col, col_span=self.col_span,
            alignment=self.alignment, justification=self.justification,
        )
class Grid(Layout):
    Item = GridItem
    def __init__(self, elements,
                 rows=None, cols=None,
                 alignment=None, justification=None,
                 row_spacing=None, col_spacing=None,
                 item_attrs=None,
                 row_height='1fr',
                 column_width='1fr',
                 **attrs
                 ):
        super().__init__(elements, item_attrs=item_attrs, **attrs)
        if rows is None:
            rows = self.layout_settings['rows']
        if cols is None:
            cols = self.layout_settings['cols']
        self.rows = rows
        self.cols = cols
        self.alignment = alignment
        self.justification = justification
        self.row_gaps = row_spacing
        self.col_gaps = col_spacing
        self.row_height = row_height
        self.col_width = column_width
    def setup_layout(self, grid, attrs):
        elements = []
        nrows = 0
        ncols = 0
        for i, row in enumerate(grid):
            for j, el in enumerate(row):
                elem = self.wrap_item(el, dict(attrs, row=i+1, col=j+1))
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
        return {'rows':nrows, 'cols':ncols}, elements
    def wrap_item(self, e, attrs):
        if not isinstance(e, self.Item):
            e = self.Item(e, **attrs)
        elif hasattr(e, 'items'):
            body = e['body']
            e = dict(e)
            del e['body']
            e = GridItem(body, **dict(attrs, **e))
        else:
            if e.row is None:
                e.row = attrs['row']
            if e.col is None:
                e.col = attrs['col']
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
            if isinstance(row_height, str) and ' ' not in row_height:
                settings['grid-template-rows'] = 'repeat({rows}, {height})'.format(rows=rows, height=row_height) if isinstance(rows, int) else rows
            elif isinstance(row_height, str):
                settings['grid-template-rows'] = row_height
            else:
                settings['grid-template-rows'] = " ".join(row_height)

        if cols is not None:
            if isinstance(col_width, str) and ' ' not in col_width:
                settings['grid-template-columns'] = 'repeat({cols}, {width})'.format(cols=cols, width=col_width) if isinstance(cols, int) else cols
            elif isinstance(row_height, str):
                settings['grid-template-columns'] = col_width
            else:
                settings['grid-template-columns'] = " ".join(col_width)
        if alignment is not None:
            settings['align-items'] = alignment
        if justification is not None:
            settings['justify-items'] = justification
        if row_gap is not None:
            settings['row-gap'] = row_gap
        if row_gap is not None:
            settings['column-gap'] = col_gap
        return settings
    def get_layout_styles(self):
        return self.get_grid_styles(
            rows=self.rows, cols=self.cols,
            alignment=self.alignment, justification=self.justification,
            row_gap=self.row_gaps, col_gap=self.col_gaps,
            row_height=self.row_height, col_width=self.col_width
        )

class FlexItem(LayoutItem):
    def __init__(self,
                 item,
                 order=None, grow=None,
                 shrink=None, basis=None,
                 alignment=None,
                 **attrs
                 ):
        super().__init__(item, **attrs)
        self.order = order
        self.grow = grow
        self.shrink = shrink
        self.basis = basis
        self.alignment = alignment
    @classmethod
    def get_flex_styles(cls,
                        order=None, grow=None,
                        shrink=None, basis=None,
                        alignment=None
                        ):
        settings = {}
        if order is not None:
            settings['flex-order'] = order
        if grow is not None:
            settings['flex-grow'] = grow
        if shrink is not None:
            settings['flex-shrink'] = shrink
        if basis is not None:
            settings['flex-basis'] = basis
        if alignment is not None:
            settings['align-self'] = alignment
        return settings
    def get_layout_styles(self):
        return self.get_flex_styles(
            order=self.order, grow=self.grow,
            shrink=self.shrink, basis=self.basis,
            alignment=self.alignment
        )
class Flex(Layout):
    Item = FlexItem
    def __init__(self,
                 elements,
                 direction=None, wrap=None,
                 alignment=None, justification=None,
                 content_alignment=None,
                 **attrs
                 ):
        super().__init__(elements, **attrs)
        self.direction = direction
        self.wrap = wrap
        self.content_alignment = content_alignment
        self.alignment = alignment
        self.justification = justification
    @classmethod
    def get_flex_styles(cls,
                        direction=None, wrap=None,
                        alignment=None, justification=None,
                        content_alignment=None
                        ):
        settings = {'display': 'flex'}
        if direction is not None:
            settings['flex-direction'] = direction
        if wrap is not None:
            settings['flex-wrap'] = wrap
        if alignment is not None:
            settings['align-items'] = alignment
        if justification is not None:
            settings['justify-items'] = justification
        if content_alignment is not None:
            settings['align-content'] = content_alignment
        return settings
    def get_layout_styles(self):
        return self.get_flex_styles(
            direction=self.direction, wrap=self.wrap,
            alignment=self.alignment, justification=self.justification,
            content_alignment=self.content_alignment
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