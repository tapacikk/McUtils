
from .HTML import HTML, CSS
from .Bootstrap import Bootstrap
from .HTMLWidgets import JupyterHTMLWrapper, HTMLWidgets
from .BootstrapWidgets import BootstrapWidgets

import functools

__all__ = [ "JHTML" ]
__reload_hook__ = [".HTML", ".HTMLWidgets", ".Bootstrap", ".BootstrapWidgets"]

class JHTML:
    """
    Provides dispatchers to either pure HTML components or Widget components based on whether interactivity
    is required or not
    """
    @classmethod
    def load(cls):
        from IPython.core.display import display
        display(
            BootstrapWidgets.load()
            # JupyterHTMLWrapper.load_styles()
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

    def dispatcher(fn):
        name = fn.__name__
        @functools.wraps(fn)
        def dispatcher(jhtml, *elements, **attrs):
            return jhtml._dispatch(getattr(HTML, name), getattr(HTMLWidgets, name), *elements, **attrs)
        return dispatcher

    @classmethod
    @dispatcher
    def Abbr(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Address(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Anchor(jhtml, *elements, **styles): ...
    A = Anchor
    @classmethod
    @dispatcher
    def Area(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Article(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Aside(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Audio(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def B(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Base(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Bdi(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Bdo(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Blockquote(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Body(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Bold(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Br(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Button(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Canvas(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Caption(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Cite(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Code(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Col(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Colgroup(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Data(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Datalist(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dd(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Del(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Details(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dfn(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dialog(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Div(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dl(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Dt(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Em(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Embed(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Fieldset(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Figcaption(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Figure(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Footer(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Form(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Head(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Header(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Heading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Hr(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Html(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Iframe(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Image(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Img(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Input(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Ins(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Italic(jhtml, *elements, **styles): ...
    I = Italic
    @classmethod
    @dispatcher
    def Kbd(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Label(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Legend(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Link(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def List(jhtml, *elements, **styles): ...
    Ul = List
    @classmethod
    @dispatcher
    def ListItem(jhtml, *elements, **styles): ...
    Li = ListItem
    @classmethod
    @dispatcher
    def Main(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Map(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Mark(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Meta(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Meter(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Nav(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Noscript(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def NumberedList(jhtml, *elements, **styles): ...
    Ol = NumberedList
    @classmethod
    @dispatcher
    def Object(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Optgroup(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Option(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Output(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Param(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Picture(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Pre(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Progress(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Q(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Rp(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Rt(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Ruby(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def S(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Samp(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Script(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Section(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Select(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Small(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Source(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Span(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Strong(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Style(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Sub(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubHeading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubsubHeading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def SubsubsubHeading(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Summary(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Sup(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Svg(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Table(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def TableBody(jhtml, *elements, **styles): ...
    Tbody = TableBody
    @classmethod
    @dispatcher
    def TableFooter(jhtml, *elements, **styles): ...
    Tfoot = TableFooter
    @classmethod
    @dispatcher
    def TableHeader(jhtml, *elements, **styles): ...
    Thead = TableHeader
    @classmethod
    @dispatcher
    def TableHeading(jhtml, *elements, **styles): ...
    Th = TableHeading
    @classmethod
    @dispatcher
    def TableItem(jhtml, *elements, **styles): ...
    Td = TableItem
    @classmethod
    @dispatcher
    def TableRow(jhtml, *elements, **styles): ...
    Tr = TableRow
    @classmethod
    @dispatcher
    def Template(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Text(jhtml, *elements, **styles): ...
    P = Text
    @classmethod
    @dispatcher
    def Textarea(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Time(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Title(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Track(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def U(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Var(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Video(jhtml, *elements, **styles): ...
    @classmethod
    @dispatcher
    def Wbr(jhtml, *elements, **styles): ...

    del dispatcher

    class Bootstrap:

        Class = Bootstrap.Class
        Variant = Bootstrap.Variant

        @classmethod
        def _dispatch(jhtml, *elems, **attrs):
            return JHTML._dispatch(*elems, **attrs)
        def dispatcher(fn):
            name = fn.__name__
            @functools.wraps(fn)
            def dispatcher(boots, *elements, **attrs):
                return boots._dispatch(getattr(Bootstrap, name), getattr(BootstrapWidgets, name), *elements, **attrs)
            return dispatcher

        @classmethod
        @dispatcher
        def Icon(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Alert(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Badge(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def PanelBody(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def PanelHeader(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Panel(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardBody(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardHeader(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardFooter(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CardImage(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Card(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Jumbotron(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Col(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Row(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Container(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Button(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def LinkButton(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Table(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListGroup(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListGroupItem(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def FontAwesomeIcon(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def GlyphIcon(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Label(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Pill(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListComponent(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def ListItemComponent(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def Breadcrumb(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def BreadcrumbItem(boots, *elements, **styles): ...


        @classmethod
        @dispatcher
        def Accordion(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def AccordionItem(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def AccordionCollapse(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def AccordionBody(boots, *elements, **styles): ...

        @classmethod
        @dispatcher
        def Carousel(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CarouselInner(boots, *elements, **styles): ...
        @classmethod
        @dispatcher
        def CarouselItem(boots, *elements, **styles): ...

        @classmethod
        @dispatcher
        def Collapse(boots, *elements, **styles): ...

        del dispatcher