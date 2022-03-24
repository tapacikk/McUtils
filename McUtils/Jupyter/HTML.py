from xml.etree import ElementTree
import enum, weakref

__all__ = [
    "HTML",
    "Bootstrap",
    "CSS"
]

class CSS:
    """
    Defines a holder for CSS properties
    """
    def __init__(self, *selectors, **props):
        self.selectors = selectors
        self.props = self.canonicalize_props(props)
    @classmethod
    def construct(cls,
                  *selectors,
                  background=None,
                  background_attachment=None,
                  background_color=None,
                  background_image=None,
                  background_position=None,
                  background_repeat=None,
                  border=None,
                  border_bottom=None,
                  border_bottom_color=None,
                  border_bottom_style=None,
                  border_bottom_width=None,
                  border_color=None,
                  border_left=None,
                  border_left_color=None,
                  border_left_style=None,
                  border_left_width=None,
                  border_right=None,
                  border_right_color=None,
                  border_right_style=None,
                  border_right_width=None,
                  border_style=None,
                  border_top=None,
                  border_top_color=None,
                  border_top_style=None,
                  border_top_width=None,
                  border_width=None,
                  clear=None,
                  clip=None,
                  color=None,
                  cursor=None,
                  display=None,
                  filter=None,
                  float=None,
                  font=None,
                  font_family=None,
                  font_size=None,
                  font_variant=None,
                  font_weight=None,
                  height=None,
                  left=None,
                  letter_spacing=None,
                  line_height=None,
                  list_style=None,
                  list_style_image=None,
                  list_style_position=None,
                  list_style_type=None,
                  margin=None,
                  margin_bottom=None,
                  margin_left=None,
                  margin_right=None,
                  margin_top=None,
                  overflow=None,
                  padding=None,
                  padding_bottom=None,
                  padding_left=None,
                  padding_right=None,
                  padding_top=None,
                  page_break_after=None,
                  page_break_before=None,
                  position=None,
                  text_align=None,
                  text_decoration=None,
                  text_indent=None,
                  text_transform=None,
                  top=None,
                  vertical_align=None,
                  visibility=None,
                  width=None,
                  z_index=None,
                  **props
                  ):
        """
        Provides a convenience constructor for systems with autocompletions

        :param selectors:
        :type selectors:
        :param background:
        :type background:
        :param background_attachment:
        :type background_attachment:
        :param background_color:
        :type background_color:
        :param background_image:
        :type background_image:
        :param background_position:
        :type background_position:
        :param background_repeat:
        :type background_repeat:
        :param border:
        :type border:
        :param border_bottom:
        :type border_bottom:
        :param border_bottom_color:
        :type border_bottom_color:
        :param border_bottom_style:
        :type border_bottom_style:
        :param border_bottom_width:
        :type border_bottom_width:
        :param border_color:
        :type border_color:
        :param border_left:
        :type border_left:
        :param border_left_color:
        :type border_left_color:
        :param border_left_style:
        :type border_left_style:
        :param border_left_width:
        :type border_left_width:
        :param border_right:
        :type border_right:
        :param border_right_color:
        :type border_right_color:
        :param border_right_style:
        :type border_right_style:
        :param border_right_width:
        :type border_right_width:
        :param border_style:
        :type border_style:
        :param border_top:
        :type border_top:
        :param border_top_color:
        :type border_top_color:
        :param border_top_style:
        :type border_top_style:
        :param border_top_width:
        :type border_top_width:
        :param border_width:
        :type border_width:
        :param clear:
        :type clear:
        :param clip:
        :type clip:
        :param color:
        :type color:
        :param cursor:
        :type cursor:
        :param display:
        :type display:
        :param filter:
        :type filter:
        :param float:
        :type float:
        :param font:
        :type font:
        :param font_family:
        :type font_family:
        :param font_size:
        :type font_size:
        :param font_variant:
        :type font_variant:
        :param font_weight:
        :type font_weight:
        :param height:
        :type height:
        :param left:
        :type left:
        :param letter_spacing:
        :type letter_spacing:
        :param line_height:
        :type line_height:
        :param list_style:
        :type list_style:
        :param list_style_image:
        :type list_style_image:
        :param list_style_position:
        :type list_style_position:
        :param list_style_type:
        :type list_style_type:
        :param margin:
        :type margin:
        :param margin_bottom:
        :type margin_bottom:
        :param margin_left:
        :type margin_left:
        :param margin_right:
        :type margin_right:
        :param margin_top:
        :type margin_top:
        :param overflow:
        :type overflow:
        :param padding:
        :type padding:
        :param padding_bottom:
        :type padding_bottom:
        :param padding_left:
        :type padding_left:
        :param padding_right:
        :type padding_right:
        :param padding_top:
        :type padding_top:
        :param page_break_after:
        :type page_break_after:
        :param page_break_before:
        :type page_break_before:
        :param position:
        :type position:
        :param text_align:
        :type text_align:
        :param text_decoration:
        :type text_decoration:
        :param text_indent:
        :type text_indent:
        :param text_transform:
        :type text_transform:
        :param top:
        :type top:
        :param vertical_align:
        :type vertical_align:
        :param visibility:
        :type visibility:
        :param width:
        :type width:
        :param z_index:
        :type z_index:
        :param props:
        :type props:
        :return:
        :rtype:
        """
        common_props = dict(
            background=background,
            background_attachment=background_attachment,
            background_color=background_color,
            background_image=background_image,
            background_position=background_position,
            background_repeat=background_repeat,
            border=border,
            border_bottom=border_bottom,
            border_bottom_color=border_bottom_color,
            border_bottom_style=border_bottom_style,
            border_bottom_width=border_bottom_width,
            border_color=border_color,
            border_left=border_left,
            border_left_color=border_left_color,
            border_left_style=border_left_style,
            border_left_width=border_left_width,
            border_right=border_right,
            border_right_color=border_right_color,
            border_right_style=border_right_style,
            border_right_width=border_right_width,
            border_style=border_style,
            border_top=border_top,
            border_top_color=border_top_color,
            border_top_style=border_top_style,
            border_top_width=border_top_width,
            border_width=border_width,
            clear=clear,
            clip=clip,
            color=color,
            cursor=cursor,
            display=display,
            filter=filter,
            float=float,
            font=font,
            font_family=font_family,
            font_size=font_size,
            font_variant=font_variant,
            font_weight=font_weight,
            height=height,
            left=left,
            letter_spacing=letter_spacing,
            line_height=line_height,
            list_style=list_style,
            list_style_image=list_style_image,
            list_style_position=list_style_position,
            list_style_type=list_style_type,
            margin=margin,
            margin_bottom=margin_bottom,
            margin_left=margin_left,
            margin_right=margin_right,
            margin_top=margin_top,
            overflow=overflow,
            padding=padding,
            padding_bottom=padding_bottom,
            padding_left=padding_left,
            padding_right=padding_right,
            padding_top=padding_top,
            page_break_after=page_break_after,
            page_break_before=page_break_before,
            position=position,
            text_align=text_align,
            text_decoration=text_decoration,
            text_indent=text_indent,
            text_transform=text_transform,
            top=top,
            vertical_align=vertical_align,
            visibility=visibility,
            width=width,
            z_index=z_index
        )
        for k, v in common_props.items():
            if v is not None:
                props[k] = v
        return cls(*selectors, **props)
    @classmethod
    def canonicalize_props(cls, props):
        return {k.replace("_", "-"):v for k,v in props.items()}
    @classmethod
    def parse(cls, sty):
        header = sty.split("{", 1)[0]
        if len(header) == len(sty): # inline styles
            chunks = [x.strip() for x in sty.split(";")]
            splits = [x.split(":") for x in chunks if len(x) > 0]
            return cls(**{k.strip(): v.strip() for k, v in splits})
        else:
            splits = [x.split("{") for x in sty.split("}")]
            styles = []
            for key, vals in splits:
                key = [x.strip() for x in key.split(",")]
                chunks = [x.strip() for x in vals.split(";")]
                pairs = [x.split(":") for x in chunks if len(x) > 0]
                styles.append(
                    cls(*key, **{k.strip(): v.strip() for k, v in pairs})
                )
                return styles

    def tostring(self):
        if len(self.selectors) > 0:
            return "{sel} {{\n  {body}\n}}".format(
                sel=",".join(self.selectors),
                body="\n  ".join("{k}:{v};".format(k=k,v=v) for k,v in self.props.items())
            )
        else:
            return " ".join("{k}:{v};".format(k=k,v=v) for k,v in self.props.items())

class HTML:
    """
    A namespace for holding various HTML attributes
    """
    @classmethod
    def expose(cls):
        g = globals()
        for x in cls.__dict__:
            if isinstance(x, type):
                g[x.__name__] = x
    class XMLElement:
        """
        Convenience API for ElementTree
        """

        def __init__(self, tag, *elems, on_update=None, **attrs):
            self.tag = tag
            self._elems = list(elems[0] if len(elems) == 1 and isinstance(elems[0], (list, tuple)) else elems)
            self._elem_view = None
            if 'cls' in attrs:
                attrs['class'] = attrs['cls']
                del attrs['cls']
            self._attrs = attrs
            self._attr_view = None
            self._parents = weakref.WeakSet()
            self._tree_cache = None
            self.on_update = on_update if on_update is not None else lambda *s:None
        @property
        def attrs(self):
            if self._attr_view is None:
                self._attr_view = self._attrs.copy()
            return self._attr_view
        @attrs.setter
        def attrs(self, attrs):
            self._attrs = attrs
            self._attr_view = None
            self._invalidate_cache()
            self.on_update(self, 'attributes', attrs)
        @property
        def elems(self):
            if self._elem_view is None:
                self._elem_view = tuple(self._elems)
            return self._elem_view
        @elems.setter
        def elems(self, elems):
            self._elems = elems
            self._elem_view = None
            self._invalidate_cache()
            self.on_update(self)
            self.on_update(self, 'elements', elems)

        def _invalidate_cache(self):
            if self._tree_cache is not None:
                self._tree_cache = None
                for p in tuple(self._parents):
                    p._invalidate_cache()
                    self._parents.remove(p)
        def __getitem__(self, item):
            if isinstance(item, str):
                return self._attrs[item]
            else:
                return self._elems[item]
        def __setitem__(self, item, value):
            if isinstance(item, str):
                self._attrs[item] = value
                self._attr_view = None
            else:
                self._elems[item] = value
                self._elem_view = None
            self._invalidate_cache()
            self.on_update(self, item, value)

        @property
        def tree(self):
            return self.to_tree()
        def to_tree(self, root=None, parent=None):
            if parent is not None:
                self._parents.add(parent)
            if self._tree_cache is None:
                if root is None:
                    root = ElementTree.Element('root')
                attrs = self.attrs
                _copied = False
                if 'style' in attrs:
                    styles = attrs['style']
                    if hasattr(styles, 'items'):
                        styles = CSS(**styles)
                    if hasattr(styles, 'tostring'):
                        if not _copied:
                            attrs = attrs.copy()
                            _copied = True
                        attrs['style'] = styles.tostring()
                if 'class' in attrs:
                    if not isinstance(attrs['class'], str):
                        if not _copied:
                            attrs = attrs.copy()
                            _copied = True
                        try:
                            iter(attrs['class'])
                        except TypeError:
                            attrs['class'] = str(attrs['class'])
                        else:
                            attrs['class'] = " ".join(str(c) for c in attrs['class'])
                my_el = ElementTree.SubElement(root, self.tag, attrs)
                if all(isinstance(e, str) for e in self.elems):
                    my_el.text = "\n".join(self.elems)
                else:
                    for elem in self.elems:
                        if hasattr(elem, 'to_tree'):
                            elem.to_tree(root=my_el, parent=self)
                        elif hasattr(elem, 'modify'):
                            elem.modify().to_tree(root=my_el, parent=self)
                        elif isinstance(elem, ElementTree.Element):
                            my_el.append(elem)
                        elif isinstance(elem, (str, int, float, CSS)):
                            elem = str(elem)
                            kids = list(my_el)
                            if len(kids) > 0:
                                if kids[-1].tail is None:
                                    kids[-1].tail = elem
                                else:
                                    kids[-1].tail += "\n"+elem
                            else:
                                my_el.text = elem
                        elif hasattr(elem, 'to_widget'):
                            raise ValueError("can't convert {} to pure HTML. It looks like a `JupyterHTMLWrapper` so look for the appropriate `JHTML` subclass.".format(elem))
                        else:
                            raise ValueError("don't know what to do with {}".format(elem))
                self._tree_cache = my_el
            elif root is not None:
                if self._tree_cache not in root:
                    root.append(self._tree_cache)
            return self._tree_cache
        def tostring(self):
            return "\n".join(s.decode() for s in ElementTree.tostringlist(self.to_tree()))
        def __repr__(self):
            return "{}({}, {})".format(type(self).__name__, self.elems, self.attrs)
        def _repr_html_(self):
            return self.tostring()
        def make_class_list(self):
            self.attrs['class'] = self.attrs['class'].split()
        def add_class(self, *cls, copy=True):
            return HTML.ClassAdder(self, cls, copy=copy).modify()
        def remove_class(self, *cls, copy=True):
            return HTML.ClassRemover(self, cls, copy=copy).modify()
        def add_styles(self, copy=True, **sty):
            return HTML.StyleAdder(self, copy=copy, **sty).modify()

        def _find_child_node(self, etree):
            from collections import deque
            # BFS to try to find the element that matches
            remaining = deque()
            remaining.append(self)
            while remaining:
                elem = remaining.popleft()
                if isinstance(elem, HTML.ElementModifier):
                    elem = elem.modify()
                if isinstance(elem, HTML.XMLElement):
                    if etree == elem.tree:
                        return elem
                    else:
                        for e in elem.elems:
                            remaining.append(e)

        def find(self, path, find_element=True):
            base = self.tree.find(path)
            if find_element and base is not None:
                new = self._find_child_node(base)
                if new is not None:
                    base = new
            return base
        def findall(self, path, find_element=True):
            bases = self.tree.findall(path)
            if find_element:
                new = []
                for b in bases:
                    newb = self._find_child_node(b)
                    if newb is not None:
                        new.append(newb)
                    else:
                        new.append(b)
                bases = new
            return bases
        def iterfind(self, path, find_element=True):
            bases = self.tree.iterfind(path)
            for b in bases:
                if find_element:
                    newb = self._find_child_node(b)
                    if newb is not None:
                        yield newb
                    else:
                        yield b
                else:
                    yield b
        def find_by_id(self, id, mode='first', parent=None, find_element=True):
            fn = {
                'first':self.find,
                'all':self.findall,
                'iter':self.iterfind
            }[mode]
            sel = ".//*[@id='{id}']{parents}".format(id=id, parents="" if parent is None else ('.'+'.'*parent))
            return fn(sel, find_element=find_element)

        def copy(self):
            import copy
            base = copy.copy(self)
            base.attrs = base.attrs.copy()
            base._tree_cache = None
            base._parents = weakref.WeakSet()
            return base
    class ElementModifier:
        def __init__(self, my_el, copy=False):
            self.el = my_el
            self.needs_copy = copy
            self._parents = None
            self._tree_cache = None
        def modify(self):
            if self.needs_copy:
                el = self.el.copy()
            else:
                el = self.el
            return el
        def tostring(self):
            return self.modify().tostring()
        def _repr_html_(self):
            return self.tostring()
        def copy(self):
            import copy
            new = copy.copy(self)
            new.el = new.el.copy()
        def add_class(self, *cls, copy=True):
            return HTML.ClassAdder(self, cls=cls, copy=copy)
        def remove_class(self, *cls, copy=True):
            return HTML.ClassRemover(self, cls=cls, copy=copy)
        def add_styles(self, copy=True, **sty):
            return HTML.StyleAdder(self, copy=copy, **sty)
    class ClassAdder(ElementModifier):
        cls = None
        def __init__(self, el, cls=None, copy=True):
            if cls is None:
                cls = self.cls
            if isinstance(cls, str):
                cls = cls.split()
            self.cls = cls
            super().__init__(el, copy=copy)
        def modify(self):
            if isinstance(self.el, HTML.ElementModifier):
                el = self.el.modify()
            else:
                if self.needs_copy:
                    el = self.el.copy()
                else:
                    el = self.el
            if 'class' in el.attrs:
                if isinstance(el['class'], str):
                    el.make_class_list()
                class_list = list(el['class'])
                for cls in self.cls:
                    cls = str(cls)
                    if cls not in class_list:
                        class_list.append(cls)
                el['class'] = tuple(class_list)
            else:
                el['class'] = self.cls
            return el
        def __repr__(self):
            return "{}({}, {})".format(type(self).__name__, self.el, self.cls)
    class ClassRemover(ElementModifier):
        cls = None
        def __init__(self, el, cls=None, copy=True):
            if cls is None:
                cls = self.cls
            if isinstance(cls, str):
                cls = cls.split()
            self.cls = cls
            super().__init__(el, copy=copy)
        def modify(self):
            if isinstance(self.el, HTML.ElementModifier):
                el = self.el.modify()
            else:
                if self.needs_copy:
                    el = self.el.copy()
                else:
                    el = self.el
            if 'class' in el.attrs:
                if isinstance(el['class'], str):
                    el.make_class_list()
                class_list = list(el['class'])
                for cls in self.cls:
                    cls = str(cls)
                    try:
                        class_list.remove(cls)
                    except ValueError:
                        pass
                el['class'] = tuple(class_list)
            else:
                el['class'] = self.cls
            return el
        def __repr__(self):
            return "{}({}, {})".format(type(self).__name__, self.el, self.cls)
    class StyleAdder(ElementModifier):
        def __init__(self, el, copy=True, **styles):
            self.styles = styles
            super().__init__(el, copy=copy)
        def modify(self):
            if isinstance(self.el, HTML.ElementModifier):
                el = self.el.modify()
            else:
                if self.needs_copy:
                    el = self.el.copy()
                else:
                    el = self.el

            if 'style' in el.attrs:
                style = el.attrs['style']
                if isinstance(style, str):
                    style = CSS.parse(style)
                else:
                    style = style.copy()
                style.props = dict(style.props, **self.styles)
                el.attrs['style'] = style
            else:
                el.attrs['style'] = CSS(**self.styles)
            return el
        def __repr__(self):
            return "{}({}, {})".format(type(self).__name__, self.el, self.styles)

    class TagElement(XMLElement):
        tag = None
        def __init__(self, *elems, **attrs):
            super().__init__(self.tag, *elems, **attrs)
    class Nav(TagElement): tag='nav'
    class Anchor(TagElement): tag='a'
    class Text(TagElement): tag='p'
    class Div(TagElement): tag='div'
    class Heading(TagElement): tag='h1'
    class SubHeading(TagElement): tag='h2'
    class SubsubHeading(TagElement): tag='h3'
    class SubsubsubHeading(TagElement): tag='h4'
    class Small(TagElement): tag='small'
    class Bold(TagElement): tag='b'
    class Italic(TagElement): tag='i'
    class Image(TagElement): tag='img'
    class ListItem(TagElement): tag='li'
    class BaseList(TagElement):
        def __init__(self, *elems, item_attributes=None, **attrs):
            if item_attributes is None:
                item_attributes = {}
            elems = [HTML.ListItem(x, **item_attributes) if not isinstance(x, HTML.ListItem) else x for x in elems]
            super().__init__(*elems, **attrs)
    class List(BaseList): tag='ul'
    class NumberedList(BaseList): tag='ol'
    class Pre(TagElement): tag='pre'
    class Style(TagElement): tag='style'
    class Script(TagElement): tag='script'
    class Span(TagElement): tag='span'
    class Button(TagElement): tag='button'
    class TableRow(TagElement): tag='tr'
    class TableHeading(TagElement): tag='th'
    class TableItem(TagElement): tag='td'
    class Table(TagElement):
        tag = 'table'
        def __init__(self, *rows, headers=None, **attrs):
            if len(rows) == 1 and isinstance(rows[0], (list, tuple)):
                rows = rows[0]
            rows = [
                HTML.TableRow(
                    [HTML.TableItem(y) if not isinstance(y, HTML.TableItem) else y for y in x]
                ) if not isinstance(x, HTML.TableRow) else x for x in rows
            ]
            if headers is not None:
                rows = [
                    HTML.TableRow([HTML.TableHeading(x) if not isinstance(x, HTML.TableHeading) else x for x in headers])
                ] + rows
            super().__init__(rows, **attrs)

    class Canvas(TagElement): tag='canvas'

    _cls_map = None
    @classmethod
    def get_class_map(cls):
        if cls._cls_map is None:
            cls._cls_map = {}
            for v in cls.__dict__.values():
                if isinstance(v, type) and hasattr(v, 'tag'):
                    cls._cls_map[v.tag] = v
        return cls._cls_map

    @classmethod
    def convert(cls, etree:ElementTree.Element):
        children = [cls.convert(x) for x in etree]
        text = etree.text
        tail = etree.tail

        elems = (
                ([text] if text is not None else [])
                + children
                + (list(tail) if tail is not None else [])
        )

        tag = etree.tag
        map = cls.get_class_map()
        try:
            tag_class = map[tag]
        except KeyError:
            tag_class = lambda *es,**ats:HTML.XMLElement(tag, *es, **ats)

        attrs = {} if etree.attrib is None else etree.attrib

        return tag_class(*elems, **attrs)

    @classmethod
    def parse(cls, str):
        etree = ElementTree.fromstring(str)
        return cls.convert(etree)

#region Autocomplete Helpers
class SemanticVariant(enum.Enum):
    Primary = "primary"
    Secondary = "secondary"
    Danger = "danger"
    Warning = "warning"
    Info = "info"
    Dark = "dark"
    Light = "light"
    Body = "body"
    Muted = "muted"

    def __str__(self):
        return self.value

class SemanticClass(enum.Enum):
    Active='active'
    Alert = 'alert'
    AlertDanger = 'alert-danger'
    AlertDark = 'alert-dark'
    AlertDismissible = 'alert-dismissible'
    AlertHeading = 'alert-heading'
    AlertInfo = 'alert-info'
    AlertLight = 'alert-light'
    AlertLink = 'alert-link'
    AlertPrimary = 'alert-primary'
    AlertSecondary = 'alert-secondary'
    AlertSuccess = 'alert-success'
    AlertWarning = 'alert-warning'
    AlignBaseline = 'align-baseline'
    AlignBottom = 'align-bottom'
    AlignContentAround = 'align-content-around'
    AlignContentBetween = 'align-content-between'
    AlignContentCenter = 'align-content-center'
    AlignContentEnd = 'align-content-end'
    AlignContentLargeAround = 'align-content-lg-around'
    AlignContentLargeBetween = 'align-content-lg-between'
    AlignContentLargeCenter = 'align-content-lg-center'
    AlignContentLargeEnd = 'align-content-lg-end'
    AlignContentLargeStart = 'align-content-lg-start'
    AlignContentLargeStretch = 'align-content-lg-stretch'
    AlignContentMediumAround = 'align-content-md-around'
    AlignContentMediumBetween = 'align-content-md-between'
    AlignContentMediumCenter = 'align-content-md-center'
    AlignContentMediumEnd = 'align-content-md-end'
    AlignContentMediumStart = 'align-content-md-start'
    AlignContentMediumStretch = 'align-content-md-stretch'
    AlignContentSmallAround = 'align-content-sm-around'
    AlignContentSmallBetween = 'align-content-sm-between'
    AlignContentSmallCenter = 'align-content-sm-center'
    AlignContentSmallEnd = 'align-content-sm-end'
    AlignContentSmallStart = 'align-content-sm-start'
    AlignContentSmallStretch = 'align-content-sm-stretch'
    AlignContentStart = 'align-content-start'
    AlignContentStretch = 'align-content-stretch'
    AlignContentExtraLargeAround = 'align-content-xl-around'
    AlignContentExtraLargeBetween = 'align-content-xl-between'
    AlignContentExtraLargeCenter = 'align-content-xl-center'
    AlignContentExtraLargeEnd = 'align-content-xl-end'
    AlignContentExtraLargeStart = 'align-content-xl-start'
    AlignContentExtraLargeStretch = 'align-content-xl-stretch'
    AlignItemsBaseline = 'align-items-baseline'
    AlignItemsCenter = 'align-items-center'
    AlignItemsEnd = 'align-items-end'
    AlignItemsLargeBaseline = 'align-items-lg-baseline'
    AlignItemsLargeCenter = 'align-items-lg-center'
    AlignItemsLargeEnd = 'align-items-lg-end'
    AlignItemsLargeStart = 'align-items-lg-start'
    AlignItemsLargeStretch = 'align-items-lg-stretch'
    AlignItemsMediumBaseline = 'align-items-md-baseline'
    AlignItemsMediumCenter = 'align-items-md-center'
    AlignItemsMediumEnd = 'align-items-md-end'
    AlignItemsMediumStart = 'align-items-md-start'
    AlignItemsMediumStretch = 'align-items-md-stretch'
    AlignItemsSmallBaseline = 'align-items-sm-baseline'
    AlignItemsSmallCenter = 'align-items-sm-center'
    AlignItemsSmallEnd = 'align-items-sm-end'
    AlignItemsSmallStart = 'align-items-sm-start'
    AlignItemsSmallStretch = 'align-items-sm-stretch'
    AlignItemsStart = 'align-items-start'
    AlignItemsStretch = 'align-items-stretch'
    AlignItemsExtraLargeBaseline = 'align-items-xl-baseline'
    AlignItemsExtraLargeCenter = 'align-items-xl-center'
    AlignItemsExtraLargeEnd = 'align-items-xl-end'
    AlignItemsExtraLargeStart = 'align-items-xl-start'
    AlignItemsExtraLargeStretch = 'align-items-xl-stretch'
    AlignMiddle = 'align-middle'
    AlignSelfAuto = 'align-self-auto'
    AlignSelfBaseline = 'align-self-baseline'
    AlignSelfCenter = 'align-self-center'
    AlignSelfEnd = 'align-self-end'
    AlignSelfLargeAuto = 'align-self-lg-auto'
    AlignSelfLargeBaseline = 'align-self-lg-baseline'
    AlignSelfLargeCenter = 'align-self-lg-center'
    AlignSelfLargeEnd = 'align-self-lg-end'
    AlignSelfLargeStart = 'align-self-lg-start'
    AlignSelfLargeStretch = 'align-self-lg-stretch'
    AlignSelfMediumAuto = 'align-self-md-auto'
    AlignSelfMediumBaseline = 'align-self-md-baseline'
    AlignSelfMediumCenter = 'align-self-md-center'
    AlignSelfMediumEnd = 'align-self-md-end'
    AlignSelfMediumStart = 'align-self-md-start'
    AlignSelfMediumStretch = 'align-self-md-stretch'
    AlignSelfSmallAuto = 'align-self-sm-auto'
    AlignSelfSmallBaseline = 'align-self-sm-baseline'
    AlignSelfSmallCenter = 'align-self-sm-center'
    AlignSelfSmallEnd = 'align-self-sm-end'
    AlignSelfSmallStart = 'align-self-sm-start'
    AlignSelfSmallStretch = 'align-self-sm-stretch'
    AlignSelfStart = 'align-self-start'
    AlignSelfStretch = 'align-self-stretch'
    AlignSelfExtraLargeAuto = 'align-self-xl-auto'
    AlignSelfExtraLargeBaseline = 'align-self-xl-baseline'
    AlignSelfExtraLargeCenter = 'align-self-xl-center'
    AlignSelfExtraLargeEnd = 'align-self-xl-end'
    AlignSelfExtraLargeStart = 'align-self-xl-start'
    AlignSelfExtraLargeStretch = 'align-self-xl-stretch'
    AlignTextBottom = 'align-text-bottom'
    AlignTextTop = 'align-text-top'
    AlignTop = 'align-top'
    Arrow = 'arrow'
    Badge = 'badge'
    BadgeDanger = 'badge-danger'
    BadgeDark = 'badge-dark'
    BadgeInfo = 'badge-info'
    BadgeLight = 'badge-light'
    BadgePill = 'badge-pill'
    BadgePrimary = 'badge-primary'
    BadgeSecondary = 'badge-secondary'
    BadgeSuccess = 'badge-success'
    BadgeWarning = 'badge-warning'
    BackgroundDanger = 'bg-danger'
    BackgroundDark = 'bg-dark'
    BackgroundInfo = 'bg-info'
    BackgroundLight = 'bg-light'
    BackgroundPrimary = 'bg-primary'
    BackgroundSecondary = 'bg-secondary'
    BackgroundSuccess = 'bg-success'
    BackgroundTransparent = 'bg-transparent'
    BackgroundWarning = 'bg-warning'
    BackgroundWhite = 'bg-white'
    Blockquote = 'blockquote'
    BlockquoteFooter = 'blockquote-footer'
    Border = 'border'
    Border0 = 'border-0'
    BorderBottom = 'border-bottom'
    BorderBottom0 = 'border-bottom-0'
    BorderDanger = 'border-danger'
    BorderDark = 'border-dark'
    BorderInfo = 'border-info'
    BorderLeft = 'border-left'
    BorderLeft0 = 'border-left-0'
    BorderLight = 'border-light'
    BorderPrimary = 'border-primary'
    BorderRight = 'border-right'
    BorderRight0 = 'border-right-0'
    BorderSecondary = 'border-secondary'
    BorderSuccess = 'border-success'
    BorderTop = 'border-top'
    BorderTop0 = 'border-top-0'
    BorderWarning = 'border-warning'
    BorderWhite = 'border-white'
    Breadcrumb = 'breadcrumb'
    BreadcrumbItem = 'breadcrumb-item'
    BsPopoverBottom = 'bs-popover-bottom'
    BsPopoverLeft = 'bs-popover-left'
    BsPopoverRight = 'bs-popover-right'
    BsPopoverTop = 'bs-popover-top'
    BsTooltipBottom = 'bs-tooltip-bottom'
    BsTooltipLeft = 'bs-tooltip-left'
    BsTooltipRight = 'bs-tooltip-right'
    BsTooltipTop = 'bs-tooltip-top'
    Button = 'btn'
    ButtonBlock = 'btn-block'
    ButtonDanger = 'btn-danger'
    ButtonDark = 'btn-dark'
    ButtonGroup = 'btn-group'
    ButtonGroupVertical = 'btn-group-vertical'
    ButtonInfo = 'btn-info'
    ButtonLarge = 'btn-lg'
    ButtonLight = 'btn-light'
    ButtonLink = 'btn-link'
    ButtonOutlineDanger = 'btn-outline-danger'
    ButtonOutlineDark = 'btn-outline-dark'
    ButtonOutlineInfo = 'btn-outline-info'
    ButtonOutlineLight = 'btn-outline-light'
    ButtonOutlinePrimary = 'btn-outline-primary'
    ButtonOutlineSecondary = 'btn-outline-secondary'
    ButtonOutlineSuccess = 'btn-outline-success'
    ButtonOutlineWarning = 'btn-outline-warning'
    ButtonPrimary = 'btn-primary'
    ButtonSecondary = 'btn-secondary'
    ButtonSmall = 'btn-sm'
    ButtonSuccess = 'btn-success'
    ButtonToolbar = 'btn-toolbar'
    ButtonWarning = 'btn-warning'
    Card = 'card'
    CardBody = 'card-body'
    CardColumns = 'card-columns'
    CardDeck = 'card-deck'
    CardFooter = 'card-footer'
    CardGroup = 'card-group'
    CardHeader = 'card-header'
    CardHeaderPills = 'card-header-pills'
    CardHeaderTabs = 'card-header-tabs'
    CardImage = 'card-img'
    CardImageBottom = 'card-img-bottom'
    CardImageOverlay = 'card-img-overlay'
    CardImageTop = 'card-img-top'
    CardLink = 'card-link'
    CardSubtitle = 'card-subtitle'
    CardTitle = 'card-title'
    Carousel = 'carousel'
    CarouselCaption = 'carousel-caption'
    CarouselControlNext = 'carousel-control-next'
    CarouselControlNextIcon = 'carousel-control-next-icon'
    CarouselControlPrev = 'carousel-control-prev'
    CarouselControlPrevIcon = 'carousel-control-prev-icon'
    CarouselIndicators = 'carousel-indicators'
    CarouselInner = 'carousel-inner'
    CarouselItem = 'carousel-item'
    CarouselItemLeft = 'carousel-item-left'
    CarouselItemNext = 'carousel-item-next'
    CarouselItemPrev = 'carousel-item-prev'
    CarouselItemRight = 'carousel-item-right'
    Close = 'close'
    Column = 'col'
    Column1 = 'col-1'
    Column10 = 'col-10'
    Column11 = 'col-11'
    Column12 = 'col-12'
    Column2 = 'col-2'
    Column3 = 'col-3'
    Column4 = 'col-4'
    Column5 = 'col-5'
    Column6 = 'col-6'
    Column7 = 'col-7'
    Column8 = 'col-8'
    Column9 = 'col-9'
    ColumnAuto = 'col-auto'
    ColumnFormLabel = 'col-form-label'
    ColumnFormLabelLarge = 'col-form-label-lg'
    ColumnFormLabelSmall = 'col-form-label-sm'
    Collapsing = 'collapsing'
    ColumnLarge = 'col-lg'
    ColumnLarge1 = 'col-lg-1'
    ColumnLarge10 = 'col-lg-10'
    ColumnLarge11 = 'col-lg-11'
    ColumnLarge12 = 'col-lg-12'
    ColumnLarge2 = 'col-lg-2'
    ColumnLarge3 = 'col-lg-3'
    ColumnLarge4 = 'col-lg-4'
    ColumnLarge5 = 'col-lg-5'
    ColumnLarge6 = 'col-lg-6'
    ColumnLarge7 = 'col-lg-7'
    ColumnLarge8 = 'col-lg-8'
    ColumnLarge9 = 'col-lg-9'
    ColumnLargeAuto = 'col-lg-auto'
    ColumnMedium = 'col-md'
    ColumnMedium1 = 'col-md-1'
    ColumnMedium10 = 'col-md-10'
    ColumnMedium11 = 'col-md-11'
    ColumnMedium12 = 'col-md-12'
    ColumnMedium2 = 'col-md-2'
    ColumnMedium3 = 'col-md-3'
    ColumnMedium4 = 'col-md-4'
    ColumnMedium5 = 'col-md-5'
    ColumnMedium6 = 'col-md-6'
    ColumnMedium7 = 'col-md-7'
    ColumnMedium8 = 'col-md-8'
    ColumnMedium9 = 'col-md-9'
    ColumnMediumAuto = 'col-md-auto'
    ColumnSmall = 'col-sm'
    ColumnSmall1 = 'col-sm-1'
    ColumnSmall10 = 'col-sm-10'
    ColumnSmall11 = 'col-sm-11'
    ColumnSmall12 = 'col-sm-12'
    ColumnSmall2 = 'col-sm-2'
    ColumnSmall3 = 'col-sm-3'
    ColumnSmall4 = 'col-sm-4'
    ColumnSmall5 = 'col-sm-5'
    ColumnSmall6 = 'col-sm-6'
    ColumnSmall7 = 'col-sm-7'
    ColumnSmall8 = 'col-sm-8'
    ColumnSmall9 = 'col-sm-9'
    ColumnSmallAuto = 'col-sm-auto'
    ColumnExtraLarge = 'col-xl'
    ColumnExtraLarge1 = 'col-xl-1'
    ColumnExtraLarge10 = 'col-xl-10'
    ColumnExtraLarge11 = 'col-xl-11'
    ColumnExtraLarge12 = 'col-xl-12'
    ColumnExtraLarge2 = 'col-xl-2'
    ColumnExtraLarge3 = 'col-xl-3'
    ColumnExtraLarge4 = 'col-xl-4'
    ColumnExtraLarge5 = 'col-xl-5'
    ColumnExtraLarge6 = 'col-xl-6'
    ColumnExtraLarge7 = 'col-xl-7'
    ColumnExtraLarge8 = 'col-xl-8'
    ColumnExtraLarge9 = 'col-xl-9'
    ColumnExtraLargeAuto = 'col-xl-auto'
    Container = 'container'
    ContainerFluid = 'container-fluid'
    CustomControl = 'custom-control'
    CustomControlInline = 'custom-control-inline'
    CustomControlInput = 'custom-control-input'
    CustomControlLabel = 'custom-control-label'
    CustomFile = 'custom-file'
    CustomFileInput = 'custom-file-input'
    CustomFileLabel = 'custom-file-label'
    CustomRange = 'custom-range'
    CustomSelect = 'custom-select'
    CustomSelectLarge = 'custom-select-lg'
    CustomSelectSmall = 'custom-select-sm'
    CustomSwitch = 'custom-switch'
    DisplayBlock = 'd-block'
    DisplayFlex = 'd-flex'
    DisplayInline = 'd-inline'
    DisplayInlineBlock = 'd-inline-block'
    DisplayInlineFlex = 'd-inline-flex'
    Disabled = 'disabled'
    Display1 = 'display-1'
    Display2 = 'display-2'
    Display3 = 'display-3'
    Display4 = 'display-4'
    DisplayLargeBlock = 'd-lg-block'
    DisplayLargeFlex = 'd-lg-flex'
    DisplayLargeInline = 'd-lg-inline'
    DisplayLargeInlineBlock = 'd-lg-inline-block'
    DisplayLargeInlineFlex = 'd-lg-inline-flex'
    DisplayLargeNone = 'd-lg-none'
    DisplayLargeTable = 'd-lg-table'
    DisplayLargeTableCell = 'd-lg-table-cell'
    DisplayLargeTableRow = 'd-lg-table-row'
    DisplayMediumBlock = 'd-md-block'
    DisplayMediumFlex = 'd-md-flex'
    DisplayMediumInline = 'd-md-inline'
    DisplayMediumInlineBlock = 'd-md-inline-block'
    DisplayMediumInlineFlex = 'd-md-inline-flex'
    DisplayMediumNone = 'd-md-none'
    DisplayMediumTable = 'd-md-table'
    DisplayMediumTableCell = 'd-md-table-cell'
    DisplayMediumTableRow = 'd-md-table-row'
    DisplayNone = 'd-none'
    DisplayPrintBlock = 'd-print-block'
    DisplayPrintFlex = 'd-print-flex'
    DisplayPrintInline = 'd-print-inline'
    DisplayPrintInlineBlock = 'd-print-inline-block'
    DisplayPrintInlineFlex = 'd-print-inline-flex'
    DisplayPrintNone = 'd-print-none'
    DisplayPrintTable = 'd-print-table'
    DisplayPrintTableCell = 'd-print-table-cell'
    DisplayPrintTableRow = 'd-print-table-row'
    Dropdown = 'dropdown'
    DropdownDivider = 'dropdown-divider'
    DropdownHeader = 'dropdown-header'
    DropdownItem = 'dropdown-item'
    DropdownItemText = 'dropdown-item-text'
    DropdownMenu = 'dropdown-menu'
    DropdownMenuLeft = 'dropdown-menu-left'
    DropdownMenuLargeLeft = 'dropdown-menu-lg-left'
    DropdownMenuLargeRight = 'dropdown-menu-lg-right'
    DropdownMenuMediumLeft = 'dropdown-menu-md-left'
    DropdownMenuMediumRight = 'dropdown-menu-md-right'
    DropdownMenuRight = 'dropdown-menu-right'
    DropdownMenuSmallLeft = 'dropdown-menu-sm-left'
    DropdownMenuSmallRight = 'dropdown-menu-sm-right'
    DropdownMenuExtraLargeLeft = 'dropdown-menu-xl-left'
    DropdownMenuExtraLargeRight = 'dropdown-menu-xl-right'
    DropdownToggle = 'dropdown-toggle'
    DropdownToggleSplit = 'dropdown-toggle-split'
    Dropleft = 'dropleft'
    Dropright = 'dropright'
    Dropup = 'dropup'
    DisplaySmallBlock = 'd-sm-block'
    DisplaySmallFlex = 'd-sm-flex'
    DisplaySmallInline = 'd-sm-inline'
    DisplaySmallInlineBlock = 'd-sm-inline-block'
    DisplaySmallInlineFlex = 'd-sm-inline-flex'
    DisplaySmallNone = 'd-sm-none'
    DisplaySmallTable = 'd-sm-table'
    DisplaySmallTableCell = 'd-sm-table-cell'
    DisplaySmallTableRow = 'd-sm-table-row'
    DisplayTable = 'd-table'
    DisplayTableCell = 'd-table-cell'
    DisplayTableRow = 'd-table-row'
    DisplayExtraLargeBlock = 'd-xl-block'
    DisplayExtraLargeFlex = 'd-xl-flex'
    DisplayExtraLargeInline = 'd-xl-inline'
    DisplayExtraLargeInlineBlock = 'd-xl-inline-block'
    DisplayExtraLargeInlineFlex = 'd-xl-inline-flex'
    DisplayExtraLargeNone = 'd-xl-none'
    DisplayExtraLargeTable = 'd-xl-table'
    DisplayExtraLargeTableCell = 'd-xl-table-cell'
    DisplayExtraLargeTableRow = 'd-xl-table-row'
    EmbedResponsive = 'embed-responsive'
    EmbedResponsiveItem = 'embed-responsive-item'
    Fade = 'fade'
    Figure = 'figure'
    FigureCaption = 'figure-caption'
    FigureImage = 'figure-img'
    FixedBottom = 'fixed-bottom'
    FixedTop = 'fixed-top'
    FlexColumn = 'flex-column'
    FlexColumnReverse = 'flex-column-reverse'
    FlexFill = 'flex-fill'
    FlexGrow0 = 'flex-grow-0'
    FlexGrow1 = 'flex-grow-1'
    FlexLargeColumn = 'flex-lg-column'
    FlexLargeColumnReverse = 'flex-lg-column-reverse'
    FlexLargeFill = 'flex-lg-fill'
    FlexLargeGrow0 = 'flex-lg-grow-0'
    FlexLargeGrow1 = 'flex-lg-grow-1'
    FlexLargeNowrap = 'flex-lg-nowrap'
    FlexLargeRow = 'flex-lg-row'
    FlexLargeRowReverse = 'flex-lg-row-reverse'
    FlexLargeShrink0 = 'flex-lg-shrink-0'
    FlexLargeShrink1 = 'flex-lg-shrink-1'
    FlexLargeWrap = 'flex-lg-wrap'
    FlexLargeWrapReverse = 'flex-lg-wrap-reverse'
    FlexMediumColumn = 'flex-md-column'
    FlexMediumColumnReverse = 'flex-md-column-reverse'
    FlexMediumFill = 'flex-md-fill'
    FlexMediumGrow0 = 'flex-md-grow-0'
    FlexMediumGrow1 = 'flex-md-grow-1'
    FlexMediumNowrap = 'flex-md-nowrap'
    FlexMediumRow = 'flex-md-row'
    FlexMediumRowReverse = 'flex-md-row-reverse'
    FlexMediumShrink0 = 'flex-md-shrink-0'
    FlexMediumShrink1 = 'flex-md-shrink-1'
    FlexMediumWrap = 'flex-md-wrap'
    FlexMediumWrapReverse = 'flex-md-wrap-reverse'
    FlexNowrap = 'flex-nowrap'
    FlexRow = 'flex-row'
    FlexRowReverse = 'flex-row-reverse'
    FlexShrink0 = 'flex-shrink-0'
    FlexShrink1 = 'flex-shrink-1'
    FlexSmallColumn = 'flex-sm-column'
    FlexSmallColumnReverse = 'flex-sm-column-reverse'
    FlexSmallFill = 'flex-sm-fill'
    FlexSmallGrow0 = 'flex-sm-grow-0'
    FlexSmallGrow1 = 'flex-sm-grow-1'
    FlexSmallNowrap = 'flex-sm-nowrap'
    FlexSmallRow = 'flex-sm-row'
    FlexSmallRowReverse = 'flex-sm-row-reverse'
    FlexSmallShrink0 = 'flex-sm-shrink-0'
    FlexSmallShrink1 = 'flex-sm-shrink-1'
    FlexSmallWrap = 'flex-sm-wrap'
    FlexSmallWrapReverse = 'flex-sm-wrap-reverse'
    FlexWrap = 'flex-wrap'
    FlexWrapReverse = 'flex-wrap-reverse'
    FlexExtraLargeColumn = 'flex-xl-column'
    FlexExtraLargeColumnReverse = 'flex-xl-column-reverse'
    FlexExtraLargeFill = 'flex-xl-fill'
    FlexExtraLargeGrow0 = 'flex-xl-grow-0'
    FlexExtraLargeGrow1 = 'flex-xl-grow-1'
    FlexExtraLargeNowrap = 'flex-xl-nowrap'
    FlexExtraLargeRow = 'flex-xl-row'
    FlexExtraLargeRowReverse = 'flex-xl-row-reverse'
    FlexExtraLargeShrink0 = 'flex-xl-shrink-0'
    FlexExtraLargeShrink1 = 'flex-xl-shrink-1'
    FlexExtraLargeWrap = 'flex-xl-wrap'
    FlexExtraLargeWrapReverse = 'flex-xl-wrap-reverse'
    FloatLeft = 'float-left'
    FloatLargeLeft = 'float-lg-left'
    FloatLargeNone = 'float-lg-none'
    FloatLargeRight = 'float-lg-right'
    FloatMediumLeft = 'float-md-left'
    FloatMediumNone = 'float-md-none'
    FloatMediumRight = 'float-md-right'
    FloatNone = 'float-none'
    FloatRight = 'float-right'
    FloatSmallLeft = 'float-sm-left'
    FloatSmallNone = 'float-sm-none'
    FloatSmallRight = 'float-sm-right'
    FloatExtraLargeLeft = 'float-xl-left'
    FloatExtraLargeNone = 'float-xl-none'
    FloatExtraLargeRight = 'float-xl-right'
    Focus = 'focus'
    FontItalic = 'font-italic'
    FontWeightBold = 'font-weight-bold'
    FontWeightBolder = 'font-weight-bolder'
    FontWeightLight = 'font-weight-light'
    FontWeightLighter = 'font-weight-lighter'
    FontWeightNormal = 'font-weight-normal'
    FormCheck = 'form-check'
    FormCheckInline = 'form-check-inline'
    FormCheckInput = 'form-check-input'
    FormCheckLabel = 'form-check-label'
    FormControl = 'form-control'
    FormControlFile = 'form-control-file'
    FormControlLarge = 'form-control-lg'
    FormControlPlaintext = 'form-control-plaintext'
    FormControlRange = 'form-control-range'
    FormControlSmall = 'form-control-sm'
    FormGroup = 'form-group'
    FormInline = 'form-inline'
    FormRow = 'form-row'
    FormText = 'form-text'
    H1 = 'h1'
    Height100 = 'h-100'
    H2 = 'h2'
    Height25 = 'h-25'
    H3 = 'h3'
    H4 = 'h4'
    H5 = 'h5'
    Height50 = 'h-50'
    H6 = 'h6'
    Height75 = 'h-75'
    HeightAuto = 'h-auto'
    Hide = 'hide'
    ImageFluid = 'img-fluid'
    ImageThumbnail = 'img-thumbnail'
    Initialism = 'initialism'
    InputGroup = 'input-group'
    InputGroupAppend = 'input-group-append'
    InputGroupPrepend = 'input-group-prepend'
    InputGroupText = 'input-group-text'
    InvalidFeedback = 'invalid-feedback'
    InvalidTooltip = 'invalid-tooltip'
    Invisible = 'invisible'
    IsInvalid = 'is-invalid'
    IsValid = 'is-valid'
    Jumbotron = 'jumbotron'
    JumbotronFluid = 'jumbotron-fluid'
    JustifyContentAround = 'justify-content-around'
    JustifyContentBetween = 'justify-content-between'
    JustifyContentCenter = 'justify-content-center'
    JustifyContentEnd = 'justify-content-end'
    JustifyContentLargeAround = 'justify-content-lg-around'
    JustifyContentLargeBetween = 'justify-content-lg-between'
    JustifyContentLargeCenter = 'justify-content-lg-center'
    JustifyContentLargeEnd = 'justify-content-lg-end'
    JustifyContentLargeStart = 'justify-content-lg-start'
    JustifyContentMediumAround = 'justify-content-md-around'
    JustifyContentMediumBetween = 'justify-content-md-between'
    JustifyContentMediumCenter = 'justify-content-md-center'
    JustifyContentMediumEnd = 'justify-content-md-end'
    JustifyContentMediumStart = 'justify-content-md-start'
    JustifyContentSmallAround = 'justify-content-sm-around'
    JustifyContentSmallBetween = 'justify-content-sm-between'
    JustifyContentSmallCenter = 'justify-content-sm-center'
    JustifyContentSmallEnd = 'justify-content-sm-end'
    JustifyContentSmallStart = 'justify-content-sm-start'
    JustifyContentStart = 'justify-content-start'
    JustifyContentExtraLargeAround = 'justify-content-xl-around'
    JustifyContentExtraLargeBetween = 'justify-content-xl-between'
    JustifyContentExtraLargeCenter = 'justify-content-xl-center'
    JustifyContentExtraLargeEnd = 'justify-content-xl-end'
    JustifyContentExtraLargeStart = 'justify-content-xl-start'
    Lead = 'lead'
    ListGroup = 'list-group'
    ListGroupHorizontal = 'list-group-horizontal'
    ListGroupHorizontalLarge = 'list-group-horizontal-lg'
    ListGroupHorizontalMedium = 'list-group-horizontal-md'
    ListGroupHorizontalSmall = 'list-group-horizontal-sm'
    ListGroupHorizontalExtraLarge = 'list-group-horizontal-xl'
    ListGroupItem = 'list-group-item'
    ListGroupItemAction = 'list-group-item-action'
    ListGroupItemDanger = 'list-group-item-danger'
    ListGroupItemDark = 'list-group-item-dark'
    ListGroupItemInfo = 'list-group-item-info'
    ListGroupItemLight = 'list-group-item-light'
    ListGroupItemPrimary = 'list-group-item-primary'
    ListGroupItemSecondary = 'list-group-item-secondary'
    ListGroupItemSuccess = 'list-group-item-success'
    ListGroupItemWarning = 'list-group-item-warning'
    ListInline = 'list-inline'
    ListInlineItem = 'list-inline-item'
    ListUnstyled = 'list-unstyled'
    Margin0 = 'm-0'
    Margin1 = 'm-1'
    Margin2 = 'm-2'
    Margin3 = 'm-3'
    Margin4 = 'm-4'
    Margin5 = 'm-5'
    Mark = 'mark'
    MarginAuto = 'm-auto'
    MarginBottom0 = 'mb-0'
    MarginBottom1 = 'mb-1'
    MarginBottom2 = 'mb-2'
    MarginBottom3 = 'mb-3'
    MarginBottom4 = 'mb-4'
    MarginBottom5 = 'mb-5'
    MarginBottomAuto = 'mb-auto'
    MarginBottomLarge0 = 'mb-lg-0'
    MarginBottomLarge1 = 'mb-lg-1'
    MarginBottomLarge2 = 'mb-lg-2'
    MarginBottomLarge3 = 'mb-lg-3'
    MarginBottomLarge4 = 'mb-lg-4'
    MarginBottomLarge5 = 'mb-lg-5'
    MarginBottomLargeAuto = 'mb-lg-auto'
    MarginBottomLargeN1 = 'mb-lg-n1'
    MarginBottomLargeN2 = 'mb-lg-n2'
    MarginBottomLargeN3 = 'mb-lg-n3'
    MarginBottomLargeN4 = 'mb-lg-n4'
    MarginBottomLargeN5 = 'mb-lg-n5'
    MarginBottomMedium0 = 'mb-md-0'
    MarginBottomMedium1 = 'mb-md-1'
    MarginBottomMedium2 = 'mb-md-2'
    MarginBottomMedium3 = 'mb-md-3'
    MarginBottomMedium4 = 'mb-md-4'
    MarginBottomMedium5 = 'mb-md-5'
    MarginBottomMediumAuto = 'mb-md-auto'
    MarginBottomMediumN1 = 'mb-md-n1'
    MarginBottomMediumN2 = 'mb-md-n2'
    MarginBottomMediumN3 = 'mb-md-n3'
    MarginBottomMediumN4 = 'mb-md-n4'
    MarginBottomMediumN5 = 'mb-md-n5'
    MarginBottomN1 = 'mb-n1'
    MarginBottomN2 = 'mb-n2'
    MarginBottomN3 = 'mb-n3'
    MarginBottomN4 = 'mb-n4'
    MarginBottomN5 = 'mb-n5'
    MarginBottomSmall0 = 'mb-sm-0'
    MarginBottomSmall1 = 'mb-sm-1'
    MarginBottomSmall2 = 'mb-sm-2'
    MarginBottomSmall3 = 'mb-sm-3'
    MarginBottomSmall4 = 'mb-sm-4'
    MarginBottomSmall5 = 'mb-sm-5'
    MarginBottomSmallAuto = 'mb-sm-auto'
    MarginBottomSmallN1 = 'mb-sm-n1'
    MarginBottomSmallN2 = 'mb-sm-n2'
    MarginBottomSmallN3 = 'mb-sm-n3'
    MarginBottomSmallN4 = 'mb-sm-n4'
    MarginBottomSmallN5 = 'mb-sm-n5'
    MarginBottomExtraLarge0 = 'mb-xl-0'
    MarginBottomExtraLarge1 = 'mb-xl-1'
    MarginBottomExtraLarge2 = 'mb-xl-2'
    MarginBottomExtraLarge3 = 'mb-xl-3'
    MarginBottomExtraLarge4 = 'mb-xl-4'
    MarginBottomExtraLarge5 = 'mb-xl-5'
    MarginBottomExtraLargeAuto = 'mb-xl-auto'
    MarginBottomExtraLargeN1 = 'mb-xl-n1'
    MarginBottomExtraLargeN2 = 'mb-xl-n2'
    MarginBottomExtraLargeN3 = 'mb-xl-n3'
    MarginBottomExtraLargeN4 = 'mb-xl-n4'
    MarginBottomExtraLargeN5 = 'mb-xl-n5'
    Media = 'media'
    MediaBody = 'media-body'
    Mh100 = 'mh-100'
    MinVh100 = 'min-vh-100'
    MinVw100 = 'min-vw-100'
    MarginLeft0 = 'ml-0'
    MarginLeft1 = 'ml-1'
    MarginLeft2 = 'ml-2'
    MarginLeft3 = 'ml-3'
    MarginLeft4 = 'ml-4'
    MarginLeft5 = 'ml-5'
    MarginLeftAuto = 'ml-auto'
    MarginLarge0 = 'm-lg-0'
    MarginLarge1 = 'm-lg-1'
    MarginLarge2 = 'm-lg-2'
    MarginLarge3 = 'm-lg-3'
    MarginLarge4 = 'm-lg-4'
    MarginLarge5 = 'm-lg-5'
    MarginLargeAuto = 'm-lg-auto'
    MarginLargeN1 = 'm-lg-n1'
    MarginLargeN2 = 'm-lg-n2'
    MarginLargeN3 = 'm-lg-n3'
    MarginLargeN4 = 'm-lg-n4'
    MarginLargeN5 = 'm-lg-n5'
    MarginLeftLarge0 = 'ml-lg-0'
    MarginLeftLarge1 = 'ml-lg-1'
    MarginLeftLarge2 = 'ml-lg-2'
    MarginLeftLarge3 = 'ml-lg-3'
    MarginLeftLarge4 = 'ml-lg-4'
    MarginLeftLarge5 = 'ml-lg-5'
    MarginLeftLargeAuto = 'ml-lg-auto'
    MarginLeftLargeN1 = 'ml-lg-n1'
    MarginLeftLargeN2 = 'ml-lg-n2'
    MarginLeftLargeN3 = 'ml-lg-n3'
    MarginLeftLargeN4 = 'ml-lg-n4'
    MarginLeftLargeN5 = 'ml-lg-n5'
    MarginLeftMedium0 = 'ml-md-0'
    MarginLeftMedium1 = 'ml-md-1'
    MarginLeftMedium2 = 'ml-md-2'
    MarginLeftMedium3 = 'ml-md-3'
    MarginLeftMedium4 = 'ml-md-4'
    MarginLeftMedium5 = 'ml-md-5'
    MarginLeftMediumAuto = 'ml-md-auto'
    MarginLeftMediumN1 = 'ml-md-n1'
    MarginLeftMediumN2 = 'ml-md-n2'
    MarginLeftMediumN3 = 'ml-md-n3'
    MarginLeftMediumN4 = 'ml-md-n4'
    MarginLeftMediumN5 = 'ml-md-n5'
    MarginLeftN1 = 'ml-n1'
    MarginLeftN2 = 'ml-n2'
    MarginLeftN3 = 'ml-n3'
    MarginLeftN4 = 'ml-n4'
    MarginLeftN5 = 'ml-n5'
    MarginLeftSmall0 = 'ml-sm-0'
    MarginLeftSmall1 = 'ml-sm-1'
    MarginLeftSmall2 = 'ml-sm-2'
    MarginLeftSmall3 = 'ml-sm-3'
    MarginLeftSmall4 = 'ml-sm-4'
    MarginLeftSmall5 = 'ml-sm-5'
    MarginLeftSmallAuto = 'ml-sm-auto'
    MarginLeftSmallN1 = 'ml-sm-n1'
    MarginLeftSmallN2 = 'ml-sm-n2'
    MarginLeftSmallN3 = 'ml-sm-n3'
    MarginLeftSmallN4 = 'ml-sm-n4'
    MarginLeftSmallN5 = 'ml-sm-n5'
    MarginLeftExtraLarge0 = 'ml-xl-0'
    MarginLeftExtraLarge1 = 'ml-xl-1'
    MarginLeftExtraLarge2 = 'ml-xl-2'
    MarginLeftExtraLarge3 = 'ml-xl-3'
    MarginLeftExtraLarge4 = 'ml-xl-4'
    MarginLeftExtraLarge5 = 'ml-xl-5'
    MarginLeftExtraLargeAuto = 'ml-xl-auto'
    MarginLeftExtraLargeN1 = 'ml-xl-n1'
    MarginLeftExtraLargeN2 = 'ml-xl-n2'
    MarginLeftExtraLargeN3 = 'ml-xl-n3'
    MarginLeftExtraLargeN4 = 'ml-xl-n4'
    MarginLeftExtraLargeN5 = 'ml-xl-n5'
    MarginMedium0 = 'm-md-0'
    MarginMedium1 = 'm-md-1'
    MarginMedium2 = 'm-md-2'
    MarginMedium3 = 'm-md-3'
    MarginMedium4 = 'm-md-4'
    MarginMedium5 = 'm-md-5'
    MarginMediumAuto = 'm-md-auto'
    MarginMediumN1 = 'm-md-n1'
    MarginMediumN2 = 'm-md-n2'
    MarginMediumN3 = 'm-md-n3'
    MarginMediumN4 = 'm-md-n4'
    MarginMediumN5 = 'm-md-n5'
    MarginN1 = 'm-n1'
    MarginN2 = 'm-n2'
    MarginN3 = 'm-n3'
    MarginN4 = 'm-n4'
    MarginN5 = 'm-n5'
    Modal = 'modal'
    ModalBackdrop = 'modal-backdrop'
    ModalBody = 'modal-body'
    ModalContent = 'modal-content'
    ModalDialog = 'modal-dialog'
    ModalDialogCentered = 'modal-dialog-centered'
    ModalDialogScrollable = 'modal-dialog-scrollable'
    ModalFooter = 'modal-footer'
    ModalHeader = 'modal-header'
    ModalLarge = 'modal-lg'
    ModalOpen = 'modal-open'
    ModalScrollbarMeasure = 'modal-scrollbar-measure'
    ModalSmall = 'modal-sm'
    ModalTitle = 'modal-title'
    ModalExtraLarge = 'modal-xl'
    MarginRight0 = 'mr-0'
    MarginRight1 = 'mr-1'
    MarginRight2 = 'mr-2'
    MarginRight3 = 'mr-3'
    MarginRight4 = 'mr-4'
    MarginRight5 = 'mr-5'
    MarginRightAuto = 'mr-auto'
    MarginRightLarge0 = 'mr-lg-0'
    MarginRightLarge1 = 'mr-lg-1'
    MarginRightLarge2 = 'mr-lg-2'
    MarginRightLarge3 = 'mr-lg-3'
    MarginRightLarge4 = 'mr-lg-4'
    MarginRightLarge5 = 'mr-lg-5'
    MarginRightLargeAuto = 'mr-lg-auto'
    MarginRightLargeN1 = 'mr-lg-n1'
    MarginRightLargeN2 = 'mr-lg-n2'
    MarginRightLargeN3 = 'mr-lg-n3'
    MarginRightLargeN4 = 'mr-lg-n4'
    MarginRightLargeN5 = 'mr-lg-n5'
    MarginRightMedium0 = 'mr-md-0'
    MarginRightMedium1 = 'mr-md-1'
    MarginRightMedium2 = 'mr-md-2'
    MarginRightMedium3 = 'mr-md-3'
    MarginRightMedium4 = 'mr-md-4'
    MarginRightMedium5 = 'mr-md-5'
    MarginRightMediumAuto = 'mr-md-auto'
    MarginRightMediumN1 = 'mr-md-n1'
    MarginRightMediumN2 = 'mr-md-n2'
    MarginRightMediumN3 = 'mr-md-n3'
    MarginRightMediumN4 = 'mr-md-n4'
    MarginRightMediumN5 = 'mr-md-n5'
    MarginRightN1 = 'mr-n1'
    MarginRightN2 = 'mr-n2'
    MarginRightN3 = 'mr-n3'
    MarginRightN4 = 'mr-n4'
    MarginRightN5 = 'mr-n5'
    MarginRightSmall0 = 'mr-sm-0'
    MarginRightSmall1 = 'mr-sm-1'
    MarginRightSmall2 = 'mr-sm-2'
    MarginRightSmall3 = 'mr-sm-3'
    MarginRightSmall4 = 'mr-sm-4'
    MarginRightSmall5 = 'mr-sm-5'
    MarginRightSmallAuto = 'mr-sm-auto'
    MarginRightSmallN1 = 'mr-sm-n1'
    MarginRightSmallN2 = 'mr-sm-n2'
    MarginRightSmallN3 = 'mr-sm-n3'
    MarginRightSmallN4 = 'mr-sm-n4'
    MarginRightSmallN5 = 'mr-sm-n5'
    MarginRightExtraLarge0 = 'mr-xl-0'
    MarginRightExtraLarge1 = 'mr-xl-1'
    MarginRightExtraLarge2 = 'mr-xl-2'
    MarginRightExtraLarge3 = 'mr-xl-3'
    MarginRightExtraLarge4 = 'mr-xl-4'
    MarginRightExtraLarge5 = 'mr-xl-5'
    MarginRightExtraLargeAuto = 'mr-xl-auto'
    MarginRightExtraLargeN1 = 'mr-xl-n1'
    MarginRightExtraLargeN2 = 'mr-xl-n2'
    MarginRightExtraLargeN3 = 'mr-xl-n3'
    MarginRightExtraLargeN4 = 'mr-xl-n4'
    MarginRightExtraLargeN5 = 'mr-xl-n5'
    MarginSmall0 = 'm-sm-0'
    MarginSmall1 = 'm-sm-1'
    MarginSmall2 = 'm-sm-2'
    MarginSmall3 = 'm-sm-3'
    MarginSmall4 = 'm-sm-4'
    MarginSmall5 = 'm-sm-5'
    MarginSmallAuto = 'm-sm-auto'
    MarginSmallN1 = 'm-sm-n1'
    MarginSmallN2 = 'm-sm-n2'
    MarginSmallN3 = 'm-sm-n3'
    MarginSmallN4 = 'm-sm-n4'
    MarginSmallN5 = 'm-sm-n5'
    MarginTop0 = 'mt-0'
    MarginTop1 = 'mt-1'
    MarginTop2 = 'mt-2'
    MarginTop3 = 'mt-3'
    MarginTop4 = 'mt-4'
    MarginTop5 = 'mt-5'
    MarginTopAuto = 'mt-auto'
    MarginTopLarge0 = 'mt-lg-0'
    MarginTopLarge1 = 'mt-lg-1'
    MarginTopLarge2 = 'mt-lg-2'
    MarginTopLarge3 = 'mt-lg-3'
    MarginTopLarge4 = 'mt-lg-4'
    MarginTopLarge5 = 'mt-lg-5'
    MarginTopLargeAuto = 'mt-lg-auto'
    MarginTopLargeN1 = 'mt-lg-n1'
    MarginTopLargeN2 = 'mt-lg-n2'
    MarginTopLargeN3 = 'mt-lg-n3'
    MarginTopLargeN4 = 'mt-lg-n4'
    MarginTopLargeN5 = 'mt-lg-n5'
    MarginTopMedium0 = 'mt-md-0'
    MarginTopMedium1 = 'mt-md-1'
    MarginTopMedium2 = 'mt-md-2'
    MarginTopMedium3 = 'mt-md-3'
    MarginTopMedium4 = 'mt-md-4'
    MarginTopMedium5 = 'mt-md-5'
    MarginTopMediumAuto = 'mt-md-auto'
    MarginTopMediumN1 = 'mt-md-n1'
    MarginTopMediumN2 = 'mt-md-n2'
    MarginTopMediumN3 = 'mt-md-n3'
    MarginTopMediumN4 = 'mt-md-n4'
    MarginTopMediumN5 = 'mt-md-n5'
    MarginTopN1 = 'mt-n1'
    MarginTopN2 = 'mt-n2'
    MarginTopN3 = 'mt-n3'
    MarginTopN4 = 'mt-n4'
    MarginTopN5 = 'mt-n5'
    MarginTopSmall0 = 'mt-sm-0'
    MarginTopSmall1 = 'mt-sm-1'
    MarginTopSmall2 = 'mt-sm-2'
    MarginTopSmall3 = 'mt-sm-3'
    MarginTopSmall4 = 'mt-sm-4'
    MarginTopSmall5 = 'mt-sm-5'
    MarginTopSmallAuto = 'mt-sm-auto'
    MarginTopSmallN1 = 'mt-sm-n1'
    MarginTopSmallN2 = 'mt-sm-n2'
    MarginTopSmallN3 = 'mt-sm-n3'
    MarginTopSmallN4 = 'mt-sm-n4'
    MarginTopSmallN5 = 'mt-sm-n5'
    MarginTopExtraLarge0 = 'mt-xl-0'
    MarginTopExtraLarge1 = 'mt-xl-1'
    MarginTopExtraLarge2 = 'mt-xl-2'
    MarginTopExtraLarge3 = 'mt-xl-3'
    MarginTopExtraLarge4 = 'mt-xl-4'
    MarginTopExtraLarge5 = 'mt-xl-5'
    MarginTopExtraLargeAuto = 'mt-xl-auto'
    MarginTopExtraLargeN1 = 'mt-xl-n1'
    MarginTopExtraLargeN2 = 'mt-xl-n2'
    MarginTopExtraLargeN3 = 'mt-xl-n3'
    MarginTopExtraLargeN4 = 'mt-xl-n4'
    MarginTopExtraLargeN5 = 'mt-xl-n5'
    Mw100 = 'mw-100'
    MarginX0 = 'mx-0'
    MarginX1 = 'mx-1'
    MarginX2 = 'mx-2'
    MarginX3 = 'mx-3'
    MarginX4 = 'mx-4'
    MarginX5 = 'mx-5'
    MarginXAuto = 'mx-auto'
    MarginExtraLarge0 = 'm-xl-0'
    MarginExtraLarge1 = 'm-xl-1'
    MarginExtraLarge2 = 'm-xl-2'
    MarginExtraLarge3 = 'm-xl-3'
    MarginExtraLarge4 = 'm-xl-4'
    MarginExtraLarge5 = 'm-xl-5'
    MarginExtraLargeAuto = 'm-xl-auto'
    MarginXLarge0 = 'mx-lg-0'
    MarginXLarge1 = 'mx-lg-1'
    MarginXLarge2 = 'mx-lg-2'
    MarginXLarge3 = 'mx-lg-3'
    MarginXLarge4 = 'mx-lg-4'
    MarginXLarge5 = 'mx-lg-5'
    MarginXLargeAuto = 'mx-lg-auto'
    MarginXLargeN1 = 'mx-lg-n1'
    MarginXLargeN2 = 'mx-lg-n2'
    MarginXLargeN3 = 'mx-lg-n3'
    MarginXLargeN4 = 'mx-lg-n4'
    MarginXLargeN5 = 'mx-lg-n5'
    MarginExtraLargeN1 = 'm-xl-n1'
    MarginExtraLargeN2 = 'm-xl-n2'
    MarginExtraLargeN3 = 'm-xl-n3'
    MarginExtraLargeN4 = 'm-xl-n4'
    MarginExtraLargeN5 = 'm-xl-n5'
    MarginXMedium0 = 'mx-md-0'
    MarginXMedium1 = 'mx-md-1'
    MarginXMedium2 = 'mx-md-2'
    MarginXMedium3 = 'mx-md-3'
    MarginXMedium4 = 'mx-md-4'
    MarginXMedium5 = 'mx-md-5'
    MarginXMediumAuto = 'mx-md-auto'
    MarginXMediumN1 = 'mx-md-n1'
    MarginXMediumN2 = 'mx-md-n2'
    MarginXMediumN3 = 'mx-md-n3'
    MarginXMediumN4 = 'mx-md-n4'
    MarginXMediumN5 = 'mx-md-n5'
    MarginXN1 = 'mx-n1'
    MarginXN2 = 'mx-n2'
    MarginXN3 = 'mx-n3'
    MarginXN4 = 'mx-n4'
    MarginXN5 = 'mx-n5'
    MarginXSmall0 = 'mx-sm-0'
    MarginXSmall1 = 'mx-sm-1'
    MarginXSmall2 = 'mx-sm-2'
    MarginXSmall3 = 'mx-sm-3'
    MarginXSmall4 = 'mx-sm-4'
    MarginXSmall5 = 'mx-sm-5'
    MarginXSmallAuto = 'mx-sm-auto'
    MarginXSmallN1 = 'mx-sm-n1'
    MarginXSmallN2 = 'mx-sm-n2'
    MarginXSmallN3 = 'mx-sm-n3'
    MarginXSmallN4 = 'mx-sm-n4'
    MarginXSmallN5 = 'mx-sm-n5'
    MarginXExtraLarge0 = 'mx-xl-0'
    MarginXExtraLarge1 = 'mx-xl-1'
    MarginXExtraLarge2 = 'mx-xl-2'
    MarginXExtraLarge3 = 'mx-xl-3'
    MarginXExtraLarge4 = 'mx-xl-4'
    MarginXExtraLarge5 = 'mx-xl-5'
    MarginXExtraLargeAuto = 'mx-xl-auto'
    MarginXExtraLargeN1 = 'mx-xl-n1'
    MarginXExtraLargeN2 = 'mx-xl-n2'
    MarginXExtraLargeN3 = 'mx-xl-n3'
    MarginXExtraLargeN4 = 'mx-xl-n4'
    MarginXExtraLargeN5 = 'mx-xl-n5'
    MarginY0 = 'my-0'
    MarginY1 = 'my-1'
    MarginY2 = 'my-2'
    MarginY3 = 'my-3'
    MarginY4 = 'my-4'
    MarginY5 = 'my-5'
    MarginYAuto = 'my-auto'
    MarginYLarge0 = 'my-lg-0'
    MarginYLarge1 = 'my-lg-1'
    MarginYLarge2 = 'my-lg-2'
    MarginYLarge3 = 'my-lg-3'
    MarginYLarge4 = 'my-lg-4'
    MarginYLarge5 = 'my-lg-5'
    MarginYLargeAuto = 'my-lg-auto'
    MarginYLargeN1 = 'my-lg-n1'
    MarginYLargeN2 = 'my-lg-n2'
    MarginYLargeN3 = 'my-lg-n3'
    MarginYLargeN4 = 'my-lg-n4'
    MarginYLargeN5 = 'my-lg-n5'
    MarginYMedium0 = 'my-md-0'
    MarginYMedium1 = 'my-md-1'
    MarginYMedium2 = 'my-md-2'
    MarginYMedium3 = 'my-md-3'
    MarginYMedium4 = 'my-md-4'
    MarginYMedium5 = 'my-md-5'
    MarginYMediumAuto = 'my-md-auto'
    MarginYMediumN1 = 'my-md-n1'
    MarginYMediumN2 = 'my-md-n2'
    MarginYMediumN3 = 'my-md-n3'
    MarginYMediumN4 = 'my-md-n4'
    MarginYMediumN5 = 'my-md-n5'
    MarginYN1 = 'my-n1'
    MarginYN2 = 'my-n2'
    MarginYN3 = 'my-n3'
    MarginYN4 = 'my-n4'
    MarginYN5 = 'my-n5'
    MarginYSmall0 = 'my-sm-0'
    MarginYSmall1 = 'my-sm-1'
    MarginYSmall2 = 'my-sm-2'
    MarginYSmall3 = 'my-sm-3'
    MarginYSmall4 = 'my-sm-4'
    MarginYSmall5 = 'my-sm-5'
    MarginYSmallAuto = 'my-sm-auto'
    MarginYSmallN1 = 'my-sm-n1'
    MarginYSmallN2 = 'my-sm-n2'
    MarginYSmallN3 = 'my-sm-n3'
    MarginYSmallN4 = 'my-sm-n4'
    MarginYSmallN5 = 'my-sm-n5'
    MarginYExtraLarge0 = 'my-xl-0'
    MarginYExtraLarge1 = 'my-xl-1'
    MarginYExtraLarge2 = 'my-xl-2'
    MarginYExtraLarge3 = 'my-xl-3'
    MarginYExtraLarge4 = 'my-xl-4'
    MarginYExtraLarge5 = 'my-xl-5'
    MarginYExtraLargeAuto = 'my-xl-auto'
    MarginYExtraLargeN1 = 'my-xl-n1'
    MarginYExtraLargeN2 = 'my-xl-n2'
    MarginYExtraLargeN3 = 'my-xl-n3'
    MarginYExtraLargeN4 = 'my-xl-n4'
    MarginYExtraLargeN5 = 'my-xl-n5'
    Nav = 'nav'
    Navbar = 'navbar'
    NavbarBrand = 'navbar-brand'
    NavbarCollapse = 'navbar-collapse'
    NavbarExpand = 'navbar-expand'
    NavbarExpandLarge = 'navbar-expand-lg'
    NavbarExpandMedium = 'navbar-expand-md'
    NavbarExpandSmall = 'navbar-expand-sm'
    NavbarExpandExtraLarge = 'navbar-expand-xl'
    NavbarNav = 'navbar-nav'
    NavbarText = 'navbar-text'
    NavbarToggler = 'navbar-toggler'
    NavbarTogglerIcon = 'navbar-toggler-icon'
    NavItem = 'nav-item'
    NavLink = 'nav-link'
    NavTabs = 'nav-tabs'
    NoGutters = 'no-gutters'
    Offset1 = 'offset-1'
    Offset10 = 'offset-10'
    Offset11 = 'offset-11'
    Offset2 = 'offset-2'
    Offset3 = 'offset-3'
    Offset4 = 'offset-4'
    Offset5 = 'offset-5'
    Offset6 = 'offset-6'
    Offset7 = 'offset-7'
    Offset8 = 'offset-8'
    Offset9 = 'offset-9'
    OffsetLarge0 = 'offset-lg-0'
    OffsetLarge1 = 'offset-lg-1'
    OffsetLarge10 = 'offset-lg-10'
    OffsetLarge11 = 'offset-lg-11'
    OffsetLarge2 = 'offset-lg-2'
    OffsetLarge3 = 'offset-lg-3'
    OffsetLarge4 = 'offset-lg-4'
    OffsetLarge5 = 'offset-lg-5'
    OffsetLarge6 = 'offset-lg-6'
    OffsetLarge7 = 'offset-lg-7'
    OffsetLarge8 = 'offset-lg-8'
    OffsetLarge9 = 'offset-lg-9'
    OffsetMedium0 = 'offset-md-0'
    OffsetMedium1 = 'offset-md-1'
    OffsetMedium10 = 'offset-md-10'
    OffsetMedium11 = 'offset-md-11'
    OffsetMedium2 = 'offset-md-2'
    OffsetMedium3 = 'offset-md-3'
    OffsetMedium4 = 'offset-md-4'
    OffsetMedium5 = 'offset-md-5'
    OffsetMedium6 = 'offset-md-6'
    OffsetMedium7 = 'offset-md-7'
    OffsetMedium8 = 'offset-md-8'
    OffsetMedium9 = 'offset-md-9'
    OffsetSmall0 = 'offset-sm-0'
    OffsetSmall1 = 'offset-sm-1'
    OffsetSmall10 = 'offset-sm-10'
    OffsetSmall11 = 'offset-sm-11'
    OffsetSmall2 = 'offset-sm-2'
    OffsetSmall3 = 'offset-sm-3'
    OffsetSmall4 = 'offset-sm-4'
    OffsetSmall5 = 'offset-sm-5'
    OffsetSmall6 = 'offset-sm-6'
    OffsetSmall7 = 'offset-sm-7'
    OffsetSmall8 = 'offset-sm-8'
    OffsetSmall9 = 'offset-sm-9'
    OffsetExtraLarge0 = 'offset-xl-0'
    OffsetExtraLarge1 = 'offset-xl-1'
    OffsetExtraLarge10 = 'offset-xl-10'
    OffsetExtraLarge11 = 'offset-xl-11'
    OffsetExtraLarge2 = 'offset-xl-2'
    OffsetExtraLarge3 = 'offset-xl-3'
    OffsetExtraLarge4 = 'offset-xl-4'
    OffsetExtraLarge5 = 'offset-xl-5'
    OffsetExtraLarge6 = 'offset-xl-6'
    OffsetExtraLarge7 = 'offset-xl-7'
    OffsetExtraLarge8 = 'offset-xl-8'
    OffsetExtraLarge9 = 'offset-xl-9'
    Order0 = 'order-0'
    Order1 = 'order-1'
    Order10 = 'order-10'
    Order11 = 'order-11'
    Order12 = 'order-12'
    Order2 = 'order-2'
    Order3 = 'order-3'
    Order4 = 'order-4'
    Order5 = 'order-5'
    Order6 = 'order-6'
    Order7 = 'order-7'
    Order8 = 'order-8'
    Order9 = 'order-9'
    OrderFirst = 'order-first'
    OrderLast = 'order-last'
    OrderLarge0 = 'order-lg-0'
    OrderLarge1 = 'order-lg-1'
    OrderLarge10 = 'order-lg-10'
    OrderLarge11 = 'order-lg-11'
    OrderLarge12 = 'order-lg-12'
    OrderLarge2 = 'order-lg-2'
    OrderLarge3 = 'order-lg-3'
    OrderLarge4 = 'order-lg-4'
    OrderLarge5 = 'order-lg-5'
    OrderLarge6 = 'order-lg-6'
    OrderLarge7 = 'order-lg-7'
    OrderLarge8 = 'order-lg-8'
    OrderLarge9 = 'order-lg-9'
    OrderLargeFirst = 'order-lg-first'
    OrderLargeLast = 'order-lg-last'
    OrderMedium0 = 'order-md-0'
    OrderMedium1 = 'order-md-1'
    OrderMedium10 = 'order-md-10'
    OrderMedium11 = 'order-md-11'
    OrderMedium12 = 'order-md-12'
    OrderMedium2 = 'order-md-2'
    OrderMedium3 = 'order-md-3'
    OrderMedium4 = 'order-md-4'
    OrderMedium5 = 'order-md-5'
    OrderMedium6 = 'order-md-6'
    OrderMedium7 = 'order-md-7'
    OrderMedium8 = 'order-md-8'
    OrderMedium9 = 'order-md-9'
    OrderMediumFirst = 'order-md-first'
    OrderMediumLast = 'order-md-last'
    OrderSmall0 = 'order-sm-0'
    OrderSmall1 = 'order-sm-1'
    OrderSmall10 = 'order-sm-10'
    OrderSmall11 = 'order-sm-11'
    OrderSmall12 = 'order-sm-12'
    OrderSmall2 = 'order-sm-2'
    OrderSmall3 = 'order-sm-3'
    OrderSmall4 = 'order-sm-4'
    OrderSmall5 = 'order-sm-5'
    OrderSmall6 = 'order-sm-6'
    OrderSmall7 = 'order-sm-7'
    OrderSmall8 = 'order-sm-8'
    OrderSmall9 = 'order-sm-9'
    OrderSmallFirst = 'order-sm-first'
    OrderSmallLast = 'order-sm-last'
    OrderExtraLarge0 = 'order-xl-0'
    OrderExtraLarge1 = 'order-xl-1'
    OrderExtraLarge10 = 'order-xl-10'
    OrderExtraLarge11 = 'order-xl-11'
    OrderExtraLarge12 = 'order-xl-12'
    OrderExtraLarge2 = 'order-xl-2'
    OrderExtraLarge3 = 'order-xl-3'
    OrderExtraLarge4 = 'order-xl-4'
    OrderExtraLarge5 = 'order-xl-5'
    OrderExtraLarge6 = 'order-xl-6'
    OrderExtraLarge7 = 'order-xl-7'
    OrderExtraLarge8 = 'order-xl-8'
    OrderExtraLarge9 = 'order-xl-9'
    OrderExtraLargeFirst = 'order-xl-first'
    OrderExtraLargeLast = 'order-xl-last'
    OverflowAuto = 'overflow-auto'
    OverflowHidden = 'overflow-hidden'
    Padding0 = 'p-0'
    Padding1 = 'p-1'
    Padding2 = 'p-2'
    Padding3 = 'p-3'
    Padding4 = 'p-4'
    Padding5 = 'p-5'
    PageLink = 'page-link'
    Pagination = 'pagination'
    PaddingBottom0 = 'pb-0'
    PaddingBottom1 = 'pb-1'
    PaddingBottom2 = 'pb-2'
    PaddingBottom3 = 'pb-3'
    PaddingBottom4 = 'pb-4'
    PaddingBottom5 = 'pb-5'
    PaddingBottomLarge0 = 'pb-lg-0'
    PaddingBottomLarge1 = 'pb-lg-1'
    PaddingBottomLarge2 = 'pb-lg-2'
    PaddingBottomLarge3 = 'pb-lg-3'
    PaddingBottomLarge4 = 'pb-lg-4'
    PaddingBottomLarge5 = 'pb-lg-5'
    PaddingBottomMedium0 = 'pb-md-0'
    PaddingBottomMedium1 = 'pb-md-1'
    PaddingBottomMedium2 = 'pb-md-2'
    PaddingBottomMedium3 = 'pb-md-3'
    PaddingBottomMedium4 = 'pb-md-4'
    PaddingBottomMedium5 = 'pb-md-5'
    PaddingBottomSmall0 = 'pb-sm-0'
    PaddingBottomSmall1 = 'pb-sm-1'
    PaddingBottomSmall2 = 'pb-sm-2'
    PaddingBottomSmall3 = 'pb-sm-3'
    PaddingBottomSmall4 = 'pb-sm-4'
    PaddingBottomSmall5 = 'pb-sm-5'
    PaddingBottomExtraLarge0 = 'pb-xl-0'
    PaddingBottomExtraLarge1 = 'pb-xl-1'
    PaddingBottomExtraLarge2 = 'pb-xl-2'
    PaddingBottomExtraLarge3 = 'pb-xl-3'
    PaddingBottomExtraLarge4 = 'pb-xl-4'
    PaddingBottomExtraLarge5 = 'pb-xl-5'
    PaddingLeft0 = 'pl-0'
    PaddingLeft1 = 'pl-1'
    PaddingLeft2 = 'pl-2'
    PaddingLeft3 = 'pl-3'
    PaddingLeft4 = 'pl-4'
    PaddingLeft5 = 'pl-5'
    PaddingLarge0 = 'p-lg-0'
    PaddingLarge1 = 'p-lg-1'
    PaddingLarge2 = 'p-lg-2'
    PaddingLarge3 = 'p-lg-3'
    PaddingLarge4 = 'p-lg-4'
    PaddingLarge5 = 'p-lg-5'
    PaddingLeftLarge0 = 'pl-lg-0'
    PaddingLeftLarge1 = 'pl-lg-1'
    PaddingLeftLarge2 = 'pl-lg-2'
    PaddingLeftLarge3 = 'pl-lg-3'
    PaddingLeftLarge4 = 'pl-lg-4'
    PaddingLeftLarge5 = 'pl-lg-5'
    PaddingLeftMedium0 = 'pl-md-0'
    PaddingLeftMedium1 = 'pl-md-1'
    PaddingLeftMedium2 = 'pl-md-2'
    PaddingLeftMedium3 = 'pl-md-3'
    PaddingLeftMedium4 = 'pl-md-4'
    PaddingLeftMedium5 = 'pl-md-5'
    PaddingLeftSmall0 = 'pl-sm-0'
    PaddingLeftSmall1 = 'pl-sm-1'
    PaddingLeftSmall2 = 'pl-sm-2'
    PaddingLeftSmall3 = 'pl-sm-3'
    PaddingLeftSmall4 = 'pl-sm-4'
    PaddingLeftSmall5 = 'pl-sm-5'
    PaddingLeftExtraLarge0 = 'pl-xl-0'
    PaddingLeftExtraLarge1 = 'pl-xl-1'
    PaddingLeftExtraLarge2 = 'pl-xl-2'
    PaddingLeftExtraLarge3 = 'pl-xl-3'
    PaddingLeftExtraLarge4 = 'pl-xl-4'
    PaddingLeftExtraLarge5 = 'pl-xl-5'
    PaddingMedium0 = 'p-md-0'
    PaddingMedium1 = 'p-md-1'
    PaddingMedium2 = 'p-md-2'
    PaddingMedium3 = 'p-md-3'
    PaddingMedium4 = 'p-md-4'
    PaddingMedium5 = 'p-md-5'
    PointerEvent = 'pointer-event'
    Popover = 'popover'
    PopoverBody = 'popover-body'
    PopoverHeader = 'popover-header'
    PositionAbsolute = 'position-absolute'
    PositionFixed = 'position-fixed'
    PositionRelative = 'position-relative'
    PositionStatic = 'position-static'
    PositionSticky = 'position-sticky'
    PaddingRight0 = 'pr-0'
    PaddingRight1 = 'pr-1'
    PaddingRight2 = 'pr-2'
    PaddingRight3 = 'pr-3'
    PaddingRight4 = 'pr-4'
    PaddingRight5 = 'pr-5'
    PreScrollable = 'pre-scrollable'
    PaddingRightLarge0 = 'pr-lg-0'
    PaddingRightLarge1 = 'pr-lg-1'
    PaddingRightLarge2 = 'pr-lg-2'
    PaddingRightLarge3 = 'pr-lg-3'
    PaddingRightLarge4 = 'pr-lg-4'
    PaddingRightLarge5 = 'pr-lg-5'
    PaddingRightMedium0 = 'pr-md-0'
    PaddingRightMedium1 = 'pr-md-1'
    PaddingRightMedium2 = 'pr-md-2'
    PaddingRightMedium3 = 'pr-md-3'
    PaddingRightMedium4 = 'pr-md-4'
    PaddingRightMedium5 = 'pr-md-5'
    Progress = 'progress'
    ProgressBar = 'progress-bar'
    ProgressBarAnimated = 'progress-bar-animated'
    ProgressBarStriped = 'progress-bar-striped'
    PaddingRightSmall0 = 'pr-sm-0'
    PaddingRightSmall1 = 'pr-sm-1'
    PaddingRightSmall2 = 'pr-sm-2'
    PaddingRightSmall3 = 'pr-sm-3'
    PaddingRightSmall4 = 'pr-sm-4'
    PaddingRightSmall5 = 'pr-sm-5'
    PaddingRightExtraLarge0 = 'pr-xl-0'
    PaddingRightExtraLarge1 = 'pr-xl-1'
    PaddingRightExtraLarge2 = 'pr-xl-2'
    PaddingRightExtraLarge3 = 'pr-xl-3'
    PaddingRightExtraLarge4 = 'pr-xl-4'
    PaddingRightExtraLarge5 = 'pr-xl-5'
    PaddingSmall0 = 'p-sm-0'
    PaddingSmall1 = 'p-sm-1'
    PaddingSmall2 = 'p-sm-2'
    PaddingSmall3 = 'p-sm-3'
    PaddingSmall4 = 'p-sm-4'
    PaddingSmall5 = 'p-sm-5'
    PaddingTop0 = 'pt-0'
    PaddingTop1 = 'pt-1'
    PaddingTop2 = 'pt-2'
    PaddingTop3 = 'pt-3'
    PaddingTop4 = 'pt-4'
    PaddingTop5 = 'pt-5'
    PaddingTopLarge0 = 'pt-lg-0'
    PaddingTopLarge1 = 'pt-lg-1'
    PaddingTopLarge2 = 'pt-lg-2'
    PaddingTopLarge3 = 'pt-lg-3'
    PaddingTopLarge4 = 'pt-lg-4'
    PaddingTopLarge5 = 'pt-lg-5'
    PaddingTopMedium0 = 'pt-md-0'
    PaddingTopMedium1 = 'pt-md-1'
    PaddingTopMedium2 = 'pt-md-2'
    PaddingTopMedium3 = 'pt-md-3'
    PaddingTopMedium4 = 'pt-md-4'
    PaddingTopMedium5 = 'pt-md-5'
    PaddingTopSmall0 = 'pt-sm-0'
    PaddingTopSmall1 = 'pt-sm-1'
    PaddingTopSmall2 = 'pt-sm-2'
    PaddingTopSmall3 = 'pt-sm-3'
    PaddingTopSmall4 = 'pt-sm-4'
    PaddingTopSmall5 = 'pt-sm-5'
    PaddingTopExtraLarge0 = 'pt-xl-0'
    PaddingTopExtraLarge1 = 'pt-xl-1'
    PaddingTopExtraLarge2 = 'pt-xl-2'
    PaddingTopExtraLarge3 = 'pt-xl-3'
    PaddingTopExtraLarge4 = 'pt-xl-4'
    PaddingTopExtraLarge5 = 'pt-xl-5'
    PaddingX0 = 'px-0'
    PaddingX1 = 'px-1'
    PaddingX2 = 'px-2'
    PaddingX3 = 'px-3'
    PaddingX4 = 'px-4'
    PaddingX5 = 'px-5'
    PaddingExtraLarge0 = 'p-xl-0'
    PaddingExtraLarge1 = 'p-xl-1'
    PaddingExtraLarge2 = 'p-xl-2'
    PaddingExtraLarge3 = 'p-xl-3'
    PaddingExtraLarge4 = 'p-xl-4'
    PaddingExtraLarge5 = 'p-xl-5'
    PaddingXLarge0 = 'px-lg-0'
    PaddingXLarge1 = 'px-lg-1'
    PaddingXLarge2 = 'px-lg-2'
    PaddingXLarge3 = 'px-lg-3'
    PaddingXLarge4 = 'px-lg-4'
    PaddingXLarge5 = 'px-lg-5'
    PaddingXMedium0 = 'px-md-0'
    PaddingXMedium1 = 'px-md-1'
    PaddingXMedium2 = 'px-md-2'
    PaddingXMedium3 = 'px-md-3'
    PaddingXMedium4 = 'px-md-4'
    PaddingXMedium5 = 'px-md-5'
    PaddingXSmall0 = 'px-sm-0'
    PaddingXSmall1 = 'px-sm-1'
    PaddingXSmall2 = 'px-sm-2'
    PaddingXSmall3 = 'px-sm-3'
    PaddingXSmall4 = 'px-sm-4'
    PaddingXSmall5 = 'px-sm-5'
    PaddingXExtraLarge0 = 'px-xl-0'
    PaddingXExtraLarge1 = 'px-xl-1'
    PaddingXExtraLarge2 = 'px-xl-2'
    PaddingXExtraLarge3 = 'px-xl-3'
    PaddingXExtraLarge4 = 'px-xl-4'
    PaddingXExtraLarge5 = 'px-xl-5'
    PaddingY0 = 'py-0'
    PaddingY1 = 'py-1'
    PaddingY2 = 'py-2'
    PaddingY3 = 'py-3'
    PaddingY4 = 'py-4'
    PaddingY5 = 'py-5'
    PaddingYLarge0 = 'py-lg-0'
    PaddingYLarge1 = 'py-lg-1'
    PaddingYLarge2 = 'py-lg-2'
    PaddingYLarge3 = 'py-lg-3'
    PaddingYLarge4 = 'py-lg-4'
    PaddingYLarge5 = 'py-lg-5'
    PaddingYMedium0 = 'py-md-0'
    PaddingYMedium1 = 'py-md-1'
    PaddingYMedium2 = 'py-md-2'
    PaddingYMedium3 = 'py-md-3'
    PaddingYMedium4 = 'py-md-4'
    PaddingYMedium5 = 'py-md-5'
    PaddingYSmall0 = 'py-sm-0'
    PaddingYSmall1 = 'py-sm-1'
    PaddingYSmall2 = 'py-sm-2'
    PaddingYSmall3 = 'py-sm-3'
    PaddingYSmall4 = 'py-sm-4'
    PaddingYSmall5 = 'py-sm-5'
    PaddingYExtraLarge0 = 'py-xl-0'
    PaddingYExtraLarge1 = 'py-xl-1'
    PaddingYExtraLarge2 = 'py-xl-2'
    PaddingYExtraLarge3 = 'py-xl-3'
    PaddingYExtraLarge4 = 'py-xl-4'
    PaddingYExtraLarge5 = 'py-xl-5'
    Rounded = 'rounded'
    Rounded0 = 'rounded-0'
    RoundedBottom = 'rounded-bottom'
    RoundedCircle = 'rounded-circle'
    RoundedLeft = 'rounded-left'
    RoundedLarge = 'rounded-lg'
    RoundedPill = 'rounded-pill'
    RoundedRight = 'rounded-right'
    RoundedSmall = 'rounded-sm'
    RoundedTop = 'rounded-top'
    Row = 'row'
    Shadow = 'shadow'
    ShadowLarge = 'shadow-lg'
    ShadowNone = 'shadow-none'
    ShadowSmall = 'shadow-sm'
    Show = 'show'
    Showing = 'showing'
    Small = 'small'
    SpinnerBorder = 'spinner-border'
    SpinnerBorderSmall = 'spinner-border-sm'
    SpinnerGrow = 'spinner-grow'
    SpinnerGrowSmall = 'spinner-grow-sm'
    SrOnly = 'sr-only'
    StickyTop = 'sticky-top'
    Table = 'table'
    TableActive = 'table-active'
    TableBordered = 'table-bordered'
    TableDanger = 'table-danger'
    TableDark = 'table-dark'
    TableInfo = 'table-info'
    TableLight = 'table-light'
    TablePrimary = 'table-primary'
    TableResponsive = 'table-responsive'
    TableResponsiveLarge = 'table-responsive-lg'
    TableResponsiveMedium = 'table-responsive-md'
    TableResponsiveSmall = 'table-responsive-sm'
    TableResponsiveExtraLarge = 'table-responsive-xl'
    TableSecondary = 'table-secondary'
    TableSuccess = 'table-success'
    TableWarning = 'table-warning'
    TabPane = 'tab-pane'
    TextBlack50 = 'text-black-50'
    TextBody = 'text-body'
    TextBreak = 'text-break'
    TextCapitalize = 'text-capitalize'
    TextCenter = 'text-center'
    TextDanger = 'text-danger'
    TextDark = 'text-dark'
    TextDecorationNone = 'text-decoration-none'
    TextHide = 'text-hide'
    TextInfo = 'text-info'
    TextJustify = 'text-justify'
    TextLeft = 'text-left'
    TextLargeCenter = 'text-lg-center'
    TextLargeLeft = 'text-lg-left'
    TextLargeRight = 'text-lg-right'
    TextLight = 'text-light'
    TextLowercase = 'text-lowercase'
    TextMediumCenter = 'text-md-center'
    TextMediumLeft = 'text-md-left'
    TextMediumRight = 'text-md-right'
    TextMonospace = 'text-monospace'
    TextMuted = 'text-muted'
    TextNowrap = 'text-nowrap'
    TextPrimary = 'text-primary'
    TextReset = 'text-reset'
    TextRight = 'text-right'
    TextSecondary = 'text-secondary'
    TextSmallCenter = 'text-sm-center'
    TextSmallLeft = 'text-sm-left'
    TextSmallRight = 'text-sm-right'
    TextSuccess = 'text-success'
    TextTruncate = 'text-truncate'
    TextUppercase = 'text-uppercase'
    TextWarning = 'text-warning'
    TextWhite = 'text-white'
    TextWhite50 = 'text-white-50'
    TextWrap = 'text-wrap'
    TextExtraLargeCenter = 'text-xl-center'
    TextExtraLargeLeft = 'text-xl-left'
    TextExtraLargeRight = 'text-xl-right'
    Toast = 'toast'
    ToastBody = 'toast-body'
    ToastHeader = 'toast-header'
    Tooltip = 'tooltip'
    TooltipInner = 'tooltip-inner'
    ValidFeedback = 'valid-feedback'
    ValidTooltip = 'valid-tooltip'
    FullHeight = 'vh-100'
    Visible = 'visible'
    FullWidth = 'vw-100'
    Width100 = 'w-100'
    Width25 = 'w-25'
    Width50 = 'w-50'
    Width75 = 'w-75'
    WidthAuto = 'w-auto'

    def __str__(self):
        return self.value
#endregion

class Bootstrap(HTML):
    # Bootstrap components
    @classmethod
    def _add_cls(kls, base_cls, cls):
        if base_cls is None:
            base_cls = cls
        else:
            if isinstance(base_cls, str):
                base_cls = base_cls.split()
            try:
                base_cls = list(base_cls)
            except TypeError:
                base_cls = [str(base_cls)]
            if isinstance(cls, str):
                cls = cls.split()
            try:
                cls = list(cls)
            except TypeError:
                cls = [str(cls)]
            for c in cls:
                if c not in base_cls:
                    base_cls.append(c)
        return base_cls
    @classmethod
    def _manage_cls(kls, obj, cls, variant):
        added_cls = obj.cls
        if variant is None:
            variant = obj.base_style
        if variant is not None:
            if isinstance(variant, (str, Bootstrap.Variant)):
                variant = [variant]
            added_cls = [obj.cls, *(obj.cls + "-" + str(v) for v in variant)]
        return Bootstrap._add_cls(cls, added_cls)

    class SpanComponent(HTML.Span):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = Bootstrap._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class Icon(SpanComponent):
        cls='glyphicon'
        def __init__(self, icon_name, **attrs):
            super().__init__(variant=icon_name, **attrs)
    class Label(SpanComponent):
        cls = 'label'
        base_style = 'default'
    class Badge(SpanComponent):
        cls = 'badge'
        base_style = None
    class Pill(SpanComponent):
        cls = ['badge', 'badge-pill']
        base_style = None

    class ListComponent(HTML.List):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = Bootstrap._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class ListItemComponent(HTML.ListItem):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = Bootstrap._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class Breadcrumb(ListComponent): cls = ['breadcrumb']
    class BreadcrumbItem(ListItemComponent): cls = ['breadcrumb-item']
    class ListGroup(ListComponent): cls='list-group'
    class ListGroupItem(ListItemComponent): cls='list-group-item'

    class DivComponent(HTML.Div):
        cls = None
        base_style = None
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = Bootstrap._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class Alert(DivComponent): cls='alert'
    class PanelBody(DivComponent): cls='panel-body'
    class PanelHeader(DivComponent): cls='panel-heading'
    class Panel(DivComponent):
        cls='panel'
        base_style='default'
        def __init__(self, *elems, header=None, **attrs):
            if header is None:
                super().__init__(*elems, **attrs)
            else:
                header = Bootstrap.PanelHeader(header)
                super().__init__(header, Bootstrap.PanelBody(*elems), **attrs)
    class CardBody(DivComponent): cls='card-body'
    class CardHeader(DivComponent): cls='card-header'
    class CardFooter(DivComponent): cls='card-footer'
    class CardImage(DivComponent): cls='card-img-top'
    class Card(DivComponent):
        cls='card'
        base_style=None
        def __init__(self, *elems, header=None, **attrs):
            if header is None:
                super().__init__(*elems, **attrs)
            else:
                header = Bootstrap.PanelHeader(header)
                super().__init__(header, Bootstrap.PanelBody(*elems), **attrs)
    class Jumbotron(DivComponent): cls='jumbotron'

    class Col(HTML.Div):
        cls='col'
        def __init__(self, *elems, width=None, size=None, cls=None, **attrs):
            added_cls = self.cls
            if size is not None:
                added_cls += '-'+str(size)
            if width is not None:
                added_cls += '-'+str(width)
            cls = Bootstrap._add_cls(cls, added_cls)
            super().__init__(*elems, cls=cls, **attrs)
        def __repr__(self):
            return "{}({})".format(type(self).__name__, self.elems)
    class Row(DivComponent):
        cls = 'row'
        def __init__(self, *cols, item_attributes=None, **attrs):
            if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
                cols = cols[0]
            if item_attributes is None:
                item_attributes = {}
            super().__init__(
                [Bootstrap.Col(x, **item_attributes) if not isinstance(x, Bootstrap.Col) else x for x in cols],
                **attrs
            )
    class Container(DivComponent): cls='container'
    @staticmethod
    def Grid(rows, row_attributes=None, item_attributes=None, auto_size=False, **attrs):
        if row_attributes is None:
            row_attributes = {}
        if item_attributes is None:
            item_attributes = {}
        if auto_size:
            # most naive approach...
            width = round(100/max(len(r) for r in rows))
            item_attributes['style'] = CSS(width=str(width)+'%')
            attrs['style'] = CSS(width='100%')
        return Bootstrap.Container(
            [Bootstrap.Row(r, item_attributes=item_attributes, **row_attributes) for r in rows],
            **attrs
        )

    class Button(HTML.Button):
        cls = 'btn'
        base_style = 'primary'
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = Bootstrap._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class LinkButton(HTML.Anchor):
        cls = 'btn'
        base_style = 'default'
        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = Bootstrap._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)
    class Table(HTML.Table):
        cls = 'table'
        base_style = 'hover'

        def __init__(self, *elems, variant=None, cls=None, **attrs):
            cls = Bootstrap._manage_cls(self, cls, variant)
            super().__init__(*elems, cls=cls, **attrs)

    Class = SemanticClass
    Variant = SemanticVariant
