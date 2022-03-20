from xml.etree import ElementTree

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

        def __init__(self, tag, *elems, **attrs):
            self.tag = tag
            self.elems = elems[0] if len(elems) == 1 and isinstance(elems[0], (list, tuple)) else elems
            if 'cls' in attrs:
                attrs['class'] = attrs['cls']
                del attrs['cls']
            self.attrs = attrs

        def to_tree(self, root=None):
            if root is None:
                root = ElementTree.Element('root')
            attrs = self.attrs
            _copied = False
            if 'style' in attrs:
                if isinstance(attrs['style'], CSS):
                    if not _copied:
                        attrs = attrs.copy()
                        _copied = True
                    attrs['style'] = attrs['style'].tostring()
            if 'class' in attrs:
                if not isinstance(attrs['class'], str):
                    if not _copied:
                        attrs = attrs.copy()
                        _copied = True
                    attrs['class'] = " ".join(attrs['class'])
            my_el = ElementTree.SubElement(root, self.tag, attrs)
            if all(isinstance(e, str) for e in self.elems):
                my_el.text = "\n".join(self.elems)
            else:
                for elem in self.elems:
                    if isinstance(elem, HTML.XMLElement):
                        elem.to_tree(root=my_el)
                    elif isinstance(elem, HTML.ElementModifier):
                        elem.modify().to_tree(root=my_el)
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
                    else:
                        raise ValueError("don't know what to do with {}".format(elem))
            return my_el
        def tostring(self):
            return "\n".join(s.decode() for s in ElementTree.tostringlist(self.to_tree()))
        def __repr__(self):
            return "{}({}, {})".format(type(self).__name__, self.elems, self.attrs)
        def _repr_html_(self):
            return self.tostring()
        def make_class_list(self):
            self.attrs['class'] = self.attrs['class'].split()
        def add_class(self, *cls):
            return HTML.ClassAdder(self, cls)
        def add_styles(self, **sty):
            return HTML.StyleAdder(self, **sty)

        def copy(self):
            import copy
            base = copy.copy(self)
            base.attrs = base.attrs.copy()
            return base
    class ElementModifier:
        def __init__(self, my_el):
            self.el = my_el
        def modify(self):
            return self.el
        def tostring(self):
            return self.modify().tostring()
        def _repr_html_(self):
            return self.tostring()
    class ClassAdder(ElementModifier):
        cls = None
        def __init__(self, el, cls=None):
            if cls is None:
                cls = self.cls
            if isinstance(cls, str):
                cls = cls.split()
            self.cls = cls
            super().__init__(el)
        def modify(self):
            el = self.el.copy()
            if 'class' in el.attrs:
                if isinstance(el.attrs['class'], str):
                    el.make_class_list()
                class_list = el.attrs['class']
                for cls in self.cls:
                    if cls not in class_list:
                        class_list.append(cls)
            else:
                el.attrs['class'] = self.cls
            return el
        def __repr__(self):
            return "{}({}, {})".format(type(self).__name__, self.el, self.cls)
    class StyleAdder(ElementModifier):
        def __init__(self, el, **styles):
            self.styles = styles
            super().__init__(el)
        def modify(self):
            el = self.el.copy()

            if 'style' in el.attrs:
                style = el.attrs['style']
                if isinstance(style, str):
                    style = CSS.parse(style)
                else:
                    style = copy.copy(style)
                style.props = dict(style.props, **self.styles)
                el.attrs['style'] = style
            else:
                el.attrs['style'] = CSS(**self.styles)
            return el
        def __repr__(self):
            return "{}({}, {})".format(type(self).__name__, self.el, self.cls)

    class TagElement(XMLElement):
        tag = None
        def __init__(self, *elems, **attrs):
            super().__init__(self.tag, *elems, **attrs)
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
                ) if not isinstance(x, HTML.TableHeading) else x for x in rows
            ]
            if headers is not None:
                rows = [
                    HTML.TableRow([HTML.TableHeading(x) if not isinstance(x, HTML.TableHeading) else x for x in headers])
                ] + rows
            super().__init__(rows, **attrs)

class Bootstrap(HTML):
    # Bootstrap components
    @classmethod
    def _add_cls(kls, base_cls, cls):
        if base_cls is None:
            base_cls = cls
        else:
            if isinstance(base_cls, str):
                base_cls = base_cls.split()
            _listed = False
            for c in base_cls:
                if c not in base_cls:
                    if not _listed:
                        base_cls = list(base_cls)
                    base_cls.append(c)
        return base_cls
    @classmethod
    def _manage_cls(kls, obj, cls, variant):
        added_cls = obj.cls
        if variant is None:
            variant = obj.base_style
        if variant is not None:
            added_cls = [obj.cls, obj.cls + "-" + variant]
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
        cls = 'label'
        base_style = None

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
        def __init__(self, *elems, width=1, size='xs', cls=None, **attrs):
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
    def Grid(rows, row_attributes=None, item_attributes=None, auto_size=True, **attrs):
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
        base_style = 'default'
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

