
from .HTML import HTML, Bootstrap, CSS
from .HTMLWidgets import JupyterHTMLWrapper, BootstrapWidgets, HTMLWidgets

__reload_hook__ = [".HTML", ".HTMLWidgets"]

__all__ = [
    "JHTML"
]

class JHTML:
    """
    Provides dispatchers to either pure HTML components or Widget components based on whether interactivity
    is required or not
    """
    @classmethod
    def load(cls):
        from IPython.core.display import display
        display(
            BootstrapWidgets.load(),
            JupyterHTMLWrapper.load_styles()
        )

    def __init__(self, context=None, include_bootstrap=False):
        self._context = context
        self._additions = set()
        self._include_boostrap = include_bootstrap
    def _get_frame_vars(self):
        import inspect
        frame = inspect.currentframe()
        return frame.f_back.f_locals
    def __enter__(self):
        """
        To make writing HTML interactively a bit nicer

        :return:
        :rtype:
        """
        globs = self._context
        if globs is None:
            globs = self._get_frame_vars()
        cls = type(self)
        for x in cls.__dict__.keys():
            if not x.startswith("_"):
                globs[x] = getattr(cls, x)
                self._additions.add(x)
        if self._include_boostrap:
            for x in cls.Bootstrap.__dict__.keys():
                if not x.startswith("_"):
                    globs[x] = getattr(cls.Bootstrap, x)
                    self._additions.add(x)
    def __exit__(self, exc_type, exc_val, exc_tb):

        globs = self._context
        if globs is None:
            globs = self._get_frame_vars()
        for x in self._additions:
            try:
                del globs[x]
            except KeyError:
                pass

    @classmethod
    def _resolve_source(jhtml, plain, widget, *elems, event_handlers=None, extra_classes=None, **attrs):
        if event_handlers is not None or extra_classes is not None:
            return widget
        else:
            Widget = JupyterHTMLWrapper.get_widgets_api().Widget
            if len(elems) > 0 and any(isinstance(e, (JupyterHTMLWrapper, Widget)) for e in elems):
                return widget
            else:
                return plain
    @classmethod
    def _dispatch(jhtml, plain, widget, *elements, event_handlers=None, extra_classes=None, **styles):
        src = jhtml._resolve_source(plain, widget, *elements, event_handlers=event_handlers, extra_classes=extra_classes, **styles)
        if src is plain:
            return src(*elements, **styles)
        else:
            return src(*elements, event_handlers=event_handlers, extra_classes=extra_classes, **styles)

    @classmethod
    def Nav(jhtml, *elements, **styles):
        plain, widget = HTML.Nav, HTMLWidgets.Nav
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Anchor(jhtml, *elements, **styles):
        plain, widget = HTML.Anchor, HTMLWidgets.Anchor
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Text(jhtml, *elements, **styles):
        plain, widget = HTML.Text, HTMLWidgets.Text
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Div(jhtml, *elements, **styles):
        plain, widget = HTML.Div, HTMLWidgets.Div
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Heading(jhtml, *elements, **styles):
        plain, widget = HTML.Heading, HTMLWidgets.Heading
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def SubHeading(jhtml, *elements, **styles):
        plain, widget = HTML.SubHeading, HTMLWidgets.SubHeading
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def SubsubHeading(jhtml, *elements, **styles):
        plain, widget = HTML.SubsubHeading, HTMLWidgets.SubsubHeading
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def SubsubsubHeading(jhtml, *elements, **styles):
        plain, widget = HTML.SubsubsubHeading, HTMLWidgets.SubsubsubHeading
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Small(jhtml, *elements, **styles):
        plain, widget = HTML.Small, HTMLWidgets.Small
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Bold(jhtml, *elements, **styles):
        plain, widget = HTML.Bold, HTMLWidgets.Bold
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Italic(jhtml, *elements, **styles):
        plain, widget = HTML.Italic, HTMLWidgets.Italic
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Image(jhtml, *elements, **styles):
        plain, widget = HTML.Image, HTMLWidgets.Image
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def ListItem(jhtml, *elements, **styles):
        plain, widget = HTML.ListItem, HTMLWidgets.ListItem
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def List(jhtml, *elements, **styles):
        plain, widget = HTML.List, HTMLWidgets.List
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def NumberedList(jhtml, *elements, **styles):
        plain, widget = HTML.NumberedList, HTMLWidgets.NumberedList
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Pre(jhtml, *elements, **styles):
        plain, widget = HTML.Pre, HTMLWidgets.Pre
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Style(jhtml, *elements, **styles):
        plain, widget = HTML.Style, HTMLWidgets.Style
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Script(jhtml, *elements, **styles):
        plain, widget = HTML.Script, HTMLWidgets.Script
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Span(jhtml, *elements, **styles):
        plain, widget = HTML.Span, HTMLWidgets.Span
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Button(jhtml, *elements, **styles):
        plain, widget = HTML.Button, HTMLWidgets.Button
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def TableRow(jhtml, *elements, **styles):
        plain, widget = HTML.TableRow, HTMLWidgets.TableRow
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def TableHeading(jhtml, *elements, **styles):
        plain, widget = HTML.TableHeading, HTMLWidgets.TableHeading
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def TableItem(jhtml, *elements, **styles):
        plain, widget = HTML.TableItem, HTMLWidgets.TableItem
        return jhtml._dispatch(plain, widget, *elements, **styles)
    @classmethod
    def Table(jhtml, *elements, **styles):
        plain, widget = HTML.Table, HTMLWidgets.Table
        return jhtml._dispatch(plain, widget, *elements, **styles)

    class Bootstrap:

        Class = Bootstrap.Class
        Variant = Bootstrap.Variant

        @classmethod
        def _dispatch(jhtml, *elems, **attrs):
            return JHTML._dispatch(*elems, **attrs)

        @classmethod
        def Icon(boots, *elements, **styles):
            plain, widget = Bootstrap.Icon, BootstrapWidgets.Icon
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Label(boots, *elements, **styles):
            plain, widget = Bootstrap.Label, BootstrapWidgets.Label
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Badge(boots, *elements, **styles):
            plain, widget = Bootstrap.Badge, BootstrapWidgets.Badge
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Pill(boots, *elements, **styles):
            plain, widget = Bootstrap.Pill, BootstrapWidgets.Pill
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Breadcrumb(boots, *elements, **styles):
            plain, widget = Bootstrap.Breadcrumb, BootstrapWidgets.Breadcrumb
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def BreadcrumbItem(boots, *elements, **styles):
            plain, widget = Bootstrap.BreadcrumbItem, BootstrapWidgets.BreadcrumbItem
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def ListGroup(boots, *elements, **styles):
            plain, widget = Bootstrap.ListGroup, BootstrapWidgets.ListGroup
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def ListGroupItem(boots, *elements, **styles):
            plain, widget = Bootstrap.ListGroupItem, BootstrapWidgets.ListGroupItem
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Alert(boots, *elements, **styles):
            plain, widget = Bootstrap.Alert, BootstrapWidgets.Alert
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def CardBody(boots, *elements, **styles):
            plain, widget = Bootstrap.CardBody, BootstrapWidgets.CardBody
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def CardHeader(boots, *elements, **styles):
            plain, widget = Bootstrap.CardHeader, BootstrapWidgets.CardHeader
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def CardFooter(boots, *elements, **styles):
            plain, widget = Bootstrap.CardFooter, BootstrapWidgets.CardFooter
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def CardImage(boots, *elements, **styles):
            plain, widget = Bootstrap.CardImage, BootstrapWidgets.CardImage
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Card(boots, *elements, **styles):
            plain, widget = Bootstrap.Card, BootstrapWidgets.Card
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Jumbotron(boots, *elements, **styles):
            plain, widget = Bootstrap.Jumbotron, BootstrapWidgets.Jumbotron
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Col(boots, *elements, **styles):
            plain, widget = Bootstrap.Col, BootstrapWidgets.Col
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Row(boots, *elements, **styles):
            plain, widget = Bootstrap.Row, BootstrapWidgets.Row
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Container(boots, *elements, **styles):
            plain, widget = Bootstrap.Container, BootstrapWidgets.Container
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Grid(boots, *elements, **styles):
            plain, widget = Bootstrap.Grid, BootstrapWidgets.Grid
            return boots._dispatch(plain, widget, *elements, **styles)

        @classmethod
        def Button(boots, *elements, **styles):
            plain, widget = Bootstrap.Button, BootstrapWidgets.Button
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def LinkButton(boots, *elements, **styles):
            plain, widget = Bootstrap.LinkButton, BootstrapWidgets.LinkButton
            return boots._dispatch(plain, widget, *elements, **styles)
        @classmethod
        def Table(boots, *elements, **styles):
            plain, widget = Bootstrap.Table, BootstrapWidgets.Table
            return boots._dispatch(plain, widget, *elements, **styles)
