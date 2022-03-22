
from .HTML import HTML, Bootstrap

__reload_hook__ = [".HTML"]

__all__ = [
    "JupyterHTMLWrapper",
    "JHTML",
    "Jupstrap"
]

class JHTMLShadowDom:
    """
    Provides a shadow DOM tree that can makes it easier to update
    chunks of a tree constructed from JupyterHTMLWrappers
    """
    def __init__(self, widg, wrapper=None, parent=None):
        self.widg = widg
        self.wrapper = wrapper
        self.parent = parent

    def to_tree(self):
        if self.parent is not None:
            return self.parent.dom.to_tree()
        else:
            if hasattr(self.widg, 'children'):
                dom_el = ...
            else:
                dom_el = HTML.parse(self.widg.value)


    @classmethod
    def _widget_parent_finder(cls, w):
        def find(selector, w=w):
            return cls._find_widget_parent(selector, w)

        return find

    @classmethod
    def to_xml_tree(cls, widg):
        ...

    @classmethod
    def _find_widget_parent(cls, selector, w, mode='tag'):
        # walk up the parent tree trying to find the tag
        try:
            parent = w.parent
        except AttributeError:
            raise ValueError("no more parents")
        else:
            if mode == 'tag':
                if parent.tag == selector:
                    return parent
                else:
                    return cls._find_widget_parent(selector, parent, mode=mode)
            elif mode == 'class':
                ...

            else:
                raise NotImplementedError("only have tag selectors for now...")


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

    cls = None #for easy overloads later
    tag = None
    def __init__(self, *elements, tag=None, event_handlers=None, layout=None, extra_classes=None):
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
        self.layout = layout
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
            x = x.tostring()
        if isinstance(x, str):
            w = self.widgets_api.HTML(x)
            w.add_class('m-0')
        elif isinstance(x, JupyterHTMLWrapper):
            w = x.to_widget(parent=self)
        else:
            w = x
        return w

    def to_widget(self, parent=None):
        layout = self.layout
        if isinstance(self.elements[0], (list, tuple)):
            if layout is None:
                layout = 'grid'
            widgets = [
                [ self._convert(x) for x in y ]
                for y in self.elements
            ]
        else:
            if layout is None:
                layout = 'column'
            widgets = [self._convert(x) for x in self.elements]
        if layout == 'grid':
            if len(widgets) == 1 and len(widgets[0]) == 1:
                widg = self.widgets_api.Box((widgets[0][0],))
            else:
                widg = self.widgets_api.GridBox(widgets)
        elif layout == 'row':
            if len(widgets) == 1:
                widg = self.widgets_api.Box((widgets[0],))
            else:
                widg = self.widgets_api.HBox(widgets)
        else:
            if len(widgets) == 1:
                widg = self.widgets_api.Box((widgets[0],))
            else:
               widg = self.widgets_api.VBox(widgets)
        if self.extra_classes is not None:
            for c in self.extra_classes:
                widg.add_class(c)
        if self.event_handlers is not None:
            listener = self.events_api.Event(source=widg, watched_events=list(self.event_handlers.keys()))
            listener.on_dom_event(self._event_handler(widg))
            widg.listener = listener

        # for discoverability in callbacks
        widg.dom = JHTMLShadowDom(widg, wrapper=self, parent=parent)
        if hasattr(widg, 'children'):
            for x in widg.children:
                if not hasattr(x, 'dom'):
                    x.dom = JHTMLShadowDom(x, wrapper=None, parent=widg)
                else:
                    x.dom.parent = widg
        return widg

    def display(self):
        return self.display_api.display(self.to_widget())

    def _ipython_display_(self):
        return self.display()

class JHTML:
    """
    Provides convenience constructors for HTML components
    """
    class WrappedElement(JupyterHTMLWrapper):
        base = None
        def __init__(self, *elems, base=None, event_handlers=None, layout=None, extra_classes=None, **attrs):
            if base is None:
                base = self.base
            self.base = base
            if any(isinstance(e, JupyterHTMLWrapper) for e in elems):
                elem = elems
                if hasattr(base, 'cls'):
                    if extra_classes is None:
                        extra_classes = base.cls
                    else:
                        extra_classes = list(extra_classes) + (
                            [base.cls] if isinstance(base.cls, str) else list(base.cls)
                        )
            else:
                elem = base(*elems, **attrs)
            super().__init__(elem, event_handlers=event_handlers, layout=layout, extra_classes=extra_classes)
        def copy(self):
            import copy
            new = copy.copy(self)
            new.elements = [x.copy() for x in new.elements]
            return new
        def add_child_class(self, *cls):
            new = self.copy()
            new.elements = [x.add_class(*cls) for x in self.elements]
            return new
        def add_class(self, *cls):
            new = self.copy()
            if new.extra_classes is None:
                new.extra_classes = list(cls)
            else:
                new.extra_classes = list(new.extra_classes)
            for c in cls:
                if c not in new.extra_classes:
                    new.extra_classes.append(c)
            return new
        def remove_class(self, *cls):
            new = self.copy()
            if new.extra_classes is None:
                new.extra_classes = list(cls)
            else:
                new.extra_classes = list(new.extra_classes)
            for c in cls:
                try:
                    new.extra_classes.remove(c)
                except ValueError:
                    pass
            return new
        def add_styles(self, **sty):
            new = self.copy()
            new.elements = [x.add_styles(**sty) for x in self.elements]
            return new
    class Anchor(WrappedElement): base=HTML.Anchor
    class Text(WrappedElement): base=HTML.Text
    class Div(WrappedElement): base=HTML.Div
    class Heading(WrappedElement): base=HTML.Heading
    class SubHeading(WrappedElement): base=HTML.SubHeading
    class SubsubHeading(WrappedElement): base=HTML.SubsubHeading
    class SubsubsubHeading(WrappedElement): base=HTML.SubsubsubHeading
    class Small(WrappedElement): base=HTML.Small
    class Bold(WrappedElement): base=HTML.Bold
    class Italic(WrappedElement): base=HTML.Italic
    class Image(WrappedElement): base=HTML.Image
    class ListItem(WrappedElement): base=HTML.Heading
    class List(WrappedElement): base=HTML.List
    class NumberedList(WrappedElement): base=HTML.NumberedList
    class Pre(WrappedElement): base=HTML.Pre
    class Style(WrappedElement): base=HTML.Style
    class Script(WrappedElement): base=HTML.Script
    class Span(WrappedElement): base=HTML.Span
    class Button(WrappedElement): base=HTML.Button
    class TableRow(WrappedElement): base=HTML.TableRow
    class TableHeading(WrappedElement): base=HTML.TableHeading
    class TableItem(WrappedElement): base=HTML.TableItem
    class Table(WrappedElement): base=HTML.Table

class Jupstrap:
    """
    Provides convenience wrappers for Bootstrap widget contexts
    """
    @classmethod
    def load(cls):
        """
        Embeds Bootstrap style definitions into the active notebook

        :return:
        :rtype:
        """
        from IPython.core.display import HTML as IPyHTML
        from urllib.request import urlopen
        return IPyHTML("<style>" + urlopen(
            'https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css').read().decode() + "</style>")

    class Alert(JHTML.WrappedElement): base = Bootstrap.Alert
    class PanelBody(JHTML.WrappedElement): base = Bootstrap.PanelBody
    class PanelHeader(JHTML.WrappedElement): base = Bootstrap.PanelHeader
    class Panel(JHTML.WrappedElement): base = Bootstrap.Panel
    class CardBody(JHTML.WrappedElement): base = Bootstrap.CardBody
    class CardHeader(JHTML.WrappedElement): base = Bootstrap.CardHeader
    class Card(JHTML.WrappedElement): base = Bootstrap.Card
    class Jumbotron(JHTML.WrappedElement): base = Bootstrap.Jumbotron
    class Col(JHTML.WrappedElement): base = Bootstrap.Col
    class Row(JHTML.WrappedElement): base = Bootstrap.Row
    class Container(JHTML.WrappedElement): base = Bootstrap.Container
    @staticmethod
    def Grid(rows, row_attributes=None, item_attributes=None, auto_size=True, **attrs):
        return JHTML.WrappedElement(rows, base=Bootstrap.Grid, row_attributes=row_attributes, item_attributes=item_attributes, auto_size=auto_size, **attrs)

    class Button(JHTML.WrappedElement): base = Bootstrap.Button
    class LinkButton(HTML.Anchor): base = Bootstrap.LinkButton
    class Table(JHTML.WrappedElement): base = Bootstrap.Table


