
from .HTML import CSS, HTML

__all__ = [
    "JupyterHTMLWrapper",
    "HTMLWidgets"
]

__reload_hook__ = [".HTML"]

class JHTMLShadowDOMElement:
    """
    Provides a shadow DOM tree that can makes it easier to update
    chunks of a tree constructed from JupyterHTMLWrappers
    """
    def __init__(self, widg, wrapper=None, parent=None):
        self.widg = widg
        self.wrapper = wrapper
        self.parent = parent.dom if hasattr(parent, 'dom') else parent
        self._tree_cache = None

    @property
    def tree(self):
        return self.to_tree()
    def to_tree(self):
        if self._tree_cache is None:
            if hasattr(self.widg, 'children'):
                children = []
                for c in self.widg.children:
                    if hasattr(c, 'dom'):
                        el = c.dom.to_tree()
                        if el is not None:
                            children.append(el)
                    elif hasattr(c, 'value'):
                        elem = HTML.parse(c.value)
                        elem.shadow = self
                        elem.on_update = self.Refresher(self, which=len(children))
                        children.append(HTML.parse(c.value))
                attrs = {}
                if len(self.widg._dom_classes) > 0:
                    attrs['cls'] = self.widg._dom_classes
                dom_el = HTML.Div(children, **attrs)
            elif hasattr(self.widg, 'value'):
                dom_el = HTML.parse(self.widg.value)
                dom_el.on_update = self.Refresher(self)
            else:
                dom_el = None
            self._tree_cache = dom_el
            if dom_el is not None:
                dom_el.shadow = self
                dom_el.on_update = self.Refresher(self)
        return self._tree_cache
    def _invalidate_cache(self):
        self._tree_cache = None
        if self.parent is not None:
            self.parent._invalidate_cache()

    class Refresher:
        def __init__(self, dom_el, which=None):
            self.dom_el = dom_el
            self.which = which
        def __call__(self, el, key, value):
            if self.which is not None:
                self.dom_el.refresh_child(el, key, value)
            else:
                self.dom_el.refresh(el, key, value)
    def refresh(self, el, key, value):
        if hasattr(self.widg, 'children'):
            if isinstance(key, str):
                if key=='elements':
                    raise NotImplementedError("haven't handle full elem updates yet")
                elif key == 'attributes':
                    k = list(value.keys())
                    if 'class' in value:
                        classes = value['class']
                        if isinstance(classes, str) or hasattr(classes, 'tostring'):
                            classes = [classes]
                        self.widg._dom_classes = [str(c) for c in classes]
                        k.remove('class')
                    if len(k) > 0:
                        raise NotImplementedError("haven't handled attr updates to keys {} yet".format(k))
                elif key == 'class':
                    if isinstance(value, str) or hasattr(value, 'tostring'):
                        value = [value]
                    self.widg._dom_classes = [str(c) for c in value]
                else:
                    raise NotImplementedError("can't update attribute {}")
            else:
                # print(self.widg.children[key])
                self.widg.children[key].value = value.tostring()
                self.widg.children[key].dom._invalidate_cache()
                # print(self.widg.children[key])
        else:
            self.widg.value = el.tostring()
        self._invalidate_cache()
    def refresh_child(self, el, key, value, which):
        raise NotImplementedError("ugh")
        subwidg = self.widg.children[which]
        if hasattr(subwidg, 'children'):
            raise NotImplementedError("ugh")
        else:
            subwidg.value = el.tostring()
        self._invalidate_cache()

    def get_parent(self, n):
        for i in range(n):
            self = self.parent
        return self

    def find(self, path, find_element=True):
        return self.tree.find(path, find_element=find_element)
    def findall(self, path, find_element=True):
        return self.tree.findall(path, find_element=find_element)
    def iterfind(self, path, find_element=True):
        return self.tree.iterfind(path, find_element=find_element)
    def find_by_id(self, id, mode='first', parent=None, find_element=True):
        return self.tree.find_by_id(id, mode=mode, parent=parent, find_element=find_element)
    def __getitem__(self, item):
        return self.to_tree()[item]

class JupyterHTMLWrapper:
    """
    Provides simple access to Jupyter display utilities
    """

    _apis = None
    @classmethod
    def load_api(cls):
        try:
            import IPython.core.display as display
        except ImportError:
            display = None
        try:
            import ipywidgets as widgets
        except ImportError:
            widgets = None
        try:
            import ipyevents as events
        except ImportError:
            events = None

        cls._apis = (
            display,
            widgets,
            events
        )

    reset_attrs = [
        # 'align-items',
        'background',
        # 'border',
        # 'border-bottom',
        # 'box-sizing',
        'color',
        # 'content',
        # 'display',
        # 'flex',
        # 'flex-direction',
        # 'flex-grow',
        'font',
        'font-family'
        # 'height',
        # 'left',
        # 'line-height',
        # 'margin-left',
        # 'min-height',
        # 'min-width',
        # 'overflow',
        # 'overflow-x',
        # 'overflow-y',
        # 'padding',
        # 'position',
        # 'top',
        # 'transform',
        # 'width'
    ]
    reset_box_attrs = [
        'display',
        'margin'
        # 'align-self',
        # 'flex-grow',
        # 'flex-shrink',
        # 'line-height',
        # 'position'
    ]
    reset_html_attrs = [
        # 'margin'
        # 'display'
        # , 'margin'
        # 'align-self',
        # 'flex-grow',
        # 'flex-shrink',
        # 'line-height',
        # 'position'
        ]
    @classmethod
    def load_styles(cls):
        """
        Embeds widget restyle definitions into the active notebook

        :return:
        :rtype:
        """
        from IPython.core.display import HTML as IPyHTML
        reset_styles = [
            CSS.construct(
                '.jupyter-widgets.jupyter-widgets-reset',
                '.jupyter-widgets-reset.jupyter-widgets',
                **{
                    k:'inherit' for k in cls.reset_attrs
                }
            ),
            CSS.construct(
                '.jupyter-widgets.widget-html',
                '.widget-html.jupyter-widgets',
                margin=0
            ),
            CSS.construct(
                '.jupyter-widgets-reset .widget-html-content',
                **{
                    k: 'inherit' for k in cls.reset_html_attrs
                }
            )
        ]
        return IPyHTML(HTML.Style("\n".join(s.tostring() for s in reset_styles)).tostring())
    stripped_classes = ['lm-Widget', 'p-Widget', 'lm-Panel', 'p-Panel', 'jupyter-widgets', 'widget-container', 'widget-inline-hbox', 'widget-box', 'widget-html']
    @classmethod
    def get_class_stripper(cls):
        import uuid
        this = "stripper"+str(uuid.uuid1()).replace("-", "")[:8]
        return cls.get_display_api().HTML(HTML.Script(
            """
var {id} = document.getElementById('{id}');
var {id}siblings = {id}.parentNode.parentNode.parentNode.childNodes;
function {id}strip(node) {{
   if (typeof node.classList != 'undefined') {{
       node.classList.remove({stripped_classes});
   }}
   if (typeof node.classList != 'undefined') {{
       for (const child of node.childNodes) {{
         {id}strip(child);
       }}
   }}
}}
for (const sibling of {id}siblings) {{
    {id}strip(sibling);
}}
{id}.parentNode.parentNode.remove();
            """.format(id=this, stripped_classes=", ".join("'{}'".format(s) for s in cls.stripped_classes)).strip(),
            id=this
        ).tostring())

    cls = None #for easy overloads later
    tag = None
    layout = None
    def __init__(self, *elements, tag=None, event_handlers=None, layout=None, extra_classes=None, **styles):
        if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
            elements = elements[0]
        self.elements = elements
        self.event_handlers = event_handlers
        if event_handlers is not None and not hasattr(self.event_handlers, 'items'):
            self.event_handlers = {
                'click':self.event_handlers,
                'mouseenter':self.event_handlers,
                'mouseleave':self.event_handlers,
                'keydown':self.event_handlers,
                'keyup':self.event_handlers
            }
        self.tag = tag if tag is not None else self.tag
        self.layout = self.layout if layout is None else layout
        self.styles = styles
        self.extra_classes = extra_classes if extra_classes is not None else self.cls
        if isinstance(self.extra_classes, str):
            self.extra_classes = [self.extra_classes]

    @staticmethod
    def _handle_event(e, event_handlers, self, widg):
        try:
            handler = event_handlers[e['type']]
        except KeyError:
            pass
        else:
            handler(e, self, widg)
    def _event_handler(self, widg):
        def handler(e):
            return self._handle_event(e, self.event_handlers, self, widg)
        return handler

    @classmethod
    def get_display_api(cls):
        if cls._apis is None:
            cls.load_api()
        return cls._apis[0]
    @property
    def display_api(self):
        return self.get_display_api()
    @classmethod
    def get_widgets_api(self):
        if self._apis is None:
            self.load_api()
        return self._apis[1]
    @property
    def widgets_api(self):
        return self.get_widgets_api()
    @classmethod
    def get_events_api(self):
        if self._apis is None:
            self.load_api()
        return self._apis[2]
    @property
    def events_api(self):
        return self.get_events_api()

    def _convert(self, x):
        if isinstance(x, (HTML.XMLElement, HTML.ElementModifier)):
            # to get widget formatting right for container classes...
            try:
                mapped_widget = self.base_map[type(x)]
            except KeyError:
                mapped_widget = None
                needs_mods = False
            else:
                try:
                    needs_mods = mapped_widget.container
                except AttributeError:
                    needs_mods = False
                if mapped_widget.cls is None:
                    mapped_widget = type(x)
                if needs_mods:
                    x = x.remove_class(mapped_widget.cls)
            x = x.tostring()
        else:
            mapped_widget = None
            needs_mods = False


        if isinstance(x, str):
            w = self.widgets_api.HTML(x)
            if needs_mods:
                extra_classes = (
                    [mapped_widget.cls]
                    if isinstance(mapped_widget.cls, str) or hasattr(mapped_widget.cls, 'tostring')
                    else list(mapped_widget.cls)
                )
                for c in extra_classes:
                    w = w.add_class(str(c))
            w.add_class('jupyter-widgets-reset')
        elif isinstance(x, JupyterHTMLWrapper):
            w = x.to_widget(parent=self)
        else:
            w = x
        return w

    def to_widget(self, parent=None):
        layout = self.layout
        if isinstance(self.elements[0], (list, tuple)):
            widgets = [
                [ self._convert(x) for x in y ]
                for y in self.elements
            ]
        else:
            widgets = [self._convert(x) for x in self.elements]
        if layout is None:
            layout = 'box'
        if layout == 'grid':
            widg = self.widgets_api.GridBox(widgets, style=self.styles)
        elif layout == 'row':
            widg = self.widgets_api.HBox(widgets, style=self.styles)
        elif layout == 'column':
            widg = self.widgets_api.VBox(widgets, style=self.styles)
        else:
            def flatten(x, base):
                if not isinstance(x, self.widgets_api.Widget):
                    for y in x:
                        flatten(y, base)
                else:
                    base.append(x)
            new = []; flatten(widgets, new)
            widg = self.widgets_api.Box(new, style=self.styles)

        widg.add_class('jupyter-widgets-reset')
        if self.extra_classes is not None:
            for c in self.extra_classes:
                widg.add_class(c)
        if self.event_handlers is not None:
            listener = self.events_api.Event(source=widg, watched_events=list(self.event_handlers.keys()))
            listener.on_dom_event(self._event_handler(widg))
            widg.listener = listener

        # for discoverability in callbacks
        widg.dom = JHTMLShadowDOMElement(widg, wrapper=self)
        if hasattr(widg, 'children'):
            for x in widg.children:
                if not hasattr(x, 'dom'):
                    x.dom = JHTMLShadowDOMElement(x, wrapper=None, parent=widg.dom)
                else:
                    x.dom.parent = widg.dom
        widg.display = lambda w=widg:self.display_widget(w)
        return widg
    @classmethod
    def display_widget(cls, w):
        return cls.get_display_api().display(w, cls.get_class_stripper())
    def display(self):
        return self.display_api.display(self.to_widget(), self.get_class_stripper())

    def _ipython_display_(self):
        return self.display()

    _widget_sources = []
    _base_map = None
    @classmethod
    def load_base_map(cls):
        if cls._base_map is None:
            cls._base_map = {}
            for src in cls._widget_sources:
                for v in src.__dict__.values():
                    if hasattr(v, 'base'):
                        cls._base_map[v.base] = v
        return cls._base_map
    @property
    def base_map(self):
        return self.load_base_map()

class HTMLWidgets:
    """
    Provides convenience constructors for HTML components
    """
    @classmethod
    def load(cls):
        return JupyterHTMLWrapper.load_styles()

    class WrappedElement(JupyterHTMLWrapper):
        base = None
        container = False
        def __init__(self, *elems, base=None, event_handlers=None, container=None, layout=None, extra_classes=None, **attrs):
            if base is None:
                base = self.base
            self.base = base
            self.container = self.container if container is None else container
            Widget = self.get_widgets_api().Widget
            if any(isinstance(e, (JupyterHTMLWrapper, Widget)) for e in elems):
                elem = elems
                if hasattr(base, 'cls'):
                    if extra_classes is None:
                        extra_classes = base.cls
                    else:
                        extra_classes = list(extra_classes) + (
                            [base.cls] if isinstance(base.cls, str) else list(base.cls)
                        )
                if hasattr(base, 'tag') and base.tag != "div":
                    raise ValueError("{} can't be a proper widget base as JupyterLab is designed (i.e. it can't have widgets as children)".format(base))
            else:
                elem = base(*elems, **attrs)
                if self.container and hasattr(base, 'cls'):
                    extras = (
                            [base.cls] if isinstance(base.cls, str) else list(base.cls)
                        )
                    if extra_classes is None:
                        extra_classes = extras
                    else:
                        extra_classes = list(extra_classes) + extras
                    elem.remove_class(*extras)
                attrs = {}
            super().__init__(elem, event_handlers=event_handlers, layout=layout, extra_classes=extra_classes, **attrs)
        def copy(self):
            import copy
            new = copy.copy(self)
            new.elements = [(x.copy() if hasattr(x, 'copy') else x) for x in new.elements]
            return new
        def add_child_class(self, *cls):
            new = self.copy()
            new.elements = [x.add_class(*cls) for x in self.elements]
            return new
        def add_class(self, *cls):
            new = self.copy()
            if new.extra_classes is None:
                new.extra_classes = [str(x) for x in cls]
            else:
                new.extra_classes = [str(x) for x in new.extra_classes]
            for c in cls:
                if c not in new.extra_classes:
                    new.extra_classes.append(str(c))
            return new
        def remove_class(self, *cls):
            new = self.copy()
            if new.extra_classes is not None:
                new.extra_classes = [str(x) for x in new.extra_classes]
            for c in cls:
                try:
                    new.extra_classes.remove(str(c))
                except ValueError:
                    pass
            return new
        def add_styles(self, **sty):
            new = self.copy()
            new.elements = [x.add_styles(**sty) for x in self.elements]
            return new
    class ContainerWrapper(WrappedElement): container = True
    class Abbr(WrappedElement): base=HTML.Abbr
    class Address(WrappedElement): base=HTML.Address
    class Anchor(WrappedElement): base=HTML.Anchor
    A = Anchor
    class Area(WrappedElement): base=HTML.Area
    class Article(WrappedElement): base=HTML.Article
    class Aside(WrappedElement): base=HTML.Aside
    class Audio(WrappedElement): base=HTML.Audio
    class B(WrappedElement): base=HTML.B
    class Base(WrappedElement): base=HTML.Base
    class BaseList(WrappedElement): base=HTML.BaseList
    class Bdi(WrappedElement): base=HTML.Bdi
    class Bdo(WrappedElement): base=HTML.Bdo
    class Blockquote(WrappedElement): base=HTML.Blockquote
    class Body(WrappedElement): base=HTML.Body
    class Bold(WrappedElement): base=HTML.Bold
    class Br(WrappedElement): base=HTML.Br
    class Button(WrappedElement): base=HTML.Button
    class Canvas(WrappedElement): base=HTML.Canvas
    class Caption(WrappedElement): base=HTML.Caption
    class Cite(WrappedElement): base=HTML.Cite
    class ClassAdder(WrappedElement): base=HTML.ClassAdder
    class ClassRemover(WrappedElement): base=HTML.ClassRemover
    class Code(WrappedElement): base=HTML.Code
    class Col(WrappedElement): base=HTML.Col
    class Colgroup(WrappedElement): base=HTML.Colgroup
    class Data(WrappedElement): base=HTML.Data
    class Datalist(WrappedElement): base=HTML.Datalist
    class Dd(WrappedElement): base=HTML.Dd
    class Del(WrappedElement): base=HTML.Del
    class Details(WrappedElement): base=HTML.Details
    class Dfn(WrappedElement): base=HTML.Dfn
    class Dialog(WrappedElement): base=HTML.Dialog
    class Div(WrappedElement): base=HTML.Div
    class Dl(WrappedElement): base=HTML.Dl
    class Dt(WrappedElement): base=HTML.Dt
    class ElementModifier(WrappedElement): base=HTML.ElementModifier
    class Em(WrappedElement): base=HTML.Em
    class Embed(WrappedElement): base=HTML.Embed
    class Fieldset(WrappedElement): base=HTML.Fieldset
    class Figcaption(WrappedElement): base=HTML.Figcaption
    class Figure(WrappedElement): base=HTML.Figure
    class Footer(WrappedElement): base=HTML.Footer
    class Form(WrappedElement): base=HTML.Form
    class Head(WrappedElement): base=HTML.Head
    class Header(WrappedElement): base=HTML.Header
    class Heading(WrappedElement): base=HTML.Heading
    class Hr(WrappedElement): base=HTML.Hr
    class Iframe(WrappedElement): base=HTML.Iframe
    class Image(WrappedElement): base=HTML.Image
    class Img(WrappedElement): base=HTML.Img
    class Input(WrappedElement): base=HTML.Input
    class Ins(WrappedElement): base=HTML.Ins
    class Italic(WrappedElement): base=HTML.Italic
    class Kbd(WrappedElement): base=HTML.Kbd
    class Label(WrappedElement): base=HTML.Label
    class Legend(WrappedElement): base=HTML.Legend
    class Li(WrappedElement): base=HTML.Li
    class Link(WrappedElement): base=HTML.Link
    class List(WrappedElement): base=HTML.List
    class ListItem(WrappedElement): base=HTML.ListItem
    class Main(WrappedElement): base=HTML.Main
    class Map(WrappedElement): base=HTML.Map
    class Mark(WrappedElement): base=HTML.Mark
    class Meta(WrappedElement): base=HTML.Meta
    class Meter(WrappedElement): base=HTML.Meter
    class Nav(WrappedElement): base=HTML.Nav
    class Noscript(WrappedElement): base=HTML.Noscript
    class NumberedList(WrappedElement): base=HTML.NumberedList
    class Object(WrappedElement): base=HTML.Object
    class Ol(WrappedElement): base=HTML.Ol
    class Optgroup(WrappedElement): base=HTML.Optgroup
    class Option(WrappedElement): base=HTML.Option
    class Output(WrappedElement): base=HTML.Output
    class P(WrappedElement): base=HTML.P
    class Param(WrappedElement): base=HTML.Param
    class Picture(WrappedElement): base=HTML.Picture
    class Pre(WrappedElement): base=HTML.Pre
    class Progress(WrappedElement): base=HTML.Progress
    class Q(WrappedElement): base=HTML.Q
    class Rp(WrappedElement): base=HTML.Rp
    class Rt(WrappedElement): base=HTML.Rt
    class Ruby(WrappedElement): base=HTML.Ruby
    class S(WrappedElement): base=HTML.S
    class Samp(WrappedElement): base=HTML.Samp
    class Script(WrappedElement): base=HTML.Script
    class Section(WrappedElement): base=HTML.Section
    class Select(WrappedElement): base=HTML.Select
    class Small(WrappedElement): base=HTML.Small
    class Source(WrappedElement): base=HTML.Source
    class Span(WrappedElement): base=HTML.Span
    class Strong(WrappedElement): base=HTML.Strong
    class Style(WrappedElement): base=HTML.Style
    class StyleAdder(WrappedElement): base=HTML.StyleAdder
    class Sub(WrappedElement): base=HTML.Sub
    class SubHeading(WrappedElement): base=HTML.SubHeading
    class SubsubHeading(WrappedElement): base=HTML.SubsubHeading
    class SubsubsubHeading(WrappedElement): base=HTML.SubsubsubHeading
    class Summary(WrappedElement): base=HTML.Summary
    class Sup(WrappedElement): base=HTML.Sup
    class Svg(WrappedElement): base=HTML.Svg
    class Table(WrappedElement): base=HTML.Table
    class TableBody(WrappedElement): base=HTML.TableBody
    class TableHeading(WrappedElement): base=HTML.TableHeading
    class TableItem(WrappedElement): base=HTML.TableItem
    class TableRow(WrappedElement): base=HTML.TableRow
    class TagElement(WrappedElement): base=HTML.TagElement
    class Tbody(WrappedElement): base=HTML.Tbody
    class Td(WrappedElement): base=HTML.Td
    class Template(WrappedElement): base=HTML.Template
    class Text(WrappedElement): base=HTML.Text
    class Textarea(WrappedElement): base=HTML.Textarea
    class Tfoot(WrappedElement): base=HTML.Tfoot
    class Th(WrappedElement): base=HTML.Th
    class Thead(WrappedElement): base=HTML.Thead
    class Time(WrappedElement): base=HTML.Time
    class Title(WrappedElement): base=HTML.Title
    class Tr(WrappedElement): base=HTML.Tr
    class Track(WrappedElement): base=HTML.Track
    class U(WrappedElement): base=HTML.U
    class Ul(WrappedElement): base=HTML.Ul
    class Var(WrappedElement): base=HTML.Var
    class Video(WrappedElement): base=HTML.Video
    class Wbr(WrappedElement): base=HTML.Wbr
JupyterHTMLWrapper._widget_sources.append(HTMLWidgets)