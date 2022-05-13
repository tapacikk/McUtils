## <a id="McUtils.Jupyter.JHTML.HTML.HTML">HTML</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L348)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L348?message=Update%20Docs)]
</div>

A namespace for holding various HTML attributes

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
keyword_replacements: dict
XMLElement: type
ElementModifier: type
ClassAdder: type
ClassRemover: type
StyleAdder: type
TagElement: type
Nav: type
Anchor: type
Text: type
Div: type
Heading: type
SubHeading: type
SubsubHeading: type
SubsubsubHeading: type
SubHeading5: type
SubHeading6: type
Small: type
Bold: type
Italic: type
Image: type
ListItem: type
BaseList: type
List: type
NumberedList: type
Pre: type
Style: type
Script: type
Span: type
Button: type
TableRow: type
TableHeading: type
TableHeader: type
TableFooter: type
TableBody: type
TableItem: type
Table: type
Canvas: type
A: type
Abbr: type
Address: type
Area: type
Article: type
Aside: type
Audio: type
B: type
Base: type
Bdi: type
Bdo: type
Blockquote: type
Body: type
Br: type
Caption: type
Cite: type
Code: type
Col: type
Colgroup: type
Data: type
Datalist: type
Dd: type
Del: type
Details: type
Dfn: type
Dialog: type
Dl: type
Dt: type
Em: type
Embed: type
Fieldset: type
Figcaption: type
Figure: type
Footer: type
Form: type
Head: type
Header: type
Hr: type
i: type
Iframe: type
Img: type
Input: type
Ins: type
Kbd: type
Label: type
Legend: type
Li: type
Link: type
Main: type
Map: type
Mark: type
Meta: type
Meter: type
Noscript: type
Object: type
Ol: type
P: type
Optgroup: type
Option: type
Output: type
Param: type
Picture: type
Progress: type
Q: type
Rp: type
Rt: type
Ruby: type
S: type
Samp: type
Section: type
Select: type
Source: type
Strong: type
Sub: type
Summary: type
Sup: type
Svg: type
Tbody: type
Td: type
Template: type
Textarea: type
Tfoot: type
Th: type
Thead: type
Time: type
Title: type
Tr: type
Track: type
U: type
Ul: type
Var: type
Video: type
Wbr: type
```
<a id="McUtils.Jupyter.JHTML.HTML.HTML.expose" class="docs-object-method">&nbsp;</a> 
```python
expose(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L352)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L352?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.manage_class" class="docs-object-method">&nbsp;</a> 
```python
manage_class(cls): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L358)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L358?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.manage_styles" class="docs-object-method">&nbsp;</a> 
```python
manage_styles(styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L372)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L372?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.clean_key" class="docs-object-method">&nbsp;</a> 
```python
clean_key(k): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L384)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L384?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.manage_attrs" class="docs-object-method">&nbsp;</a> 
```python
manage_attrs(attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L390)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L390?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.extract_styles" class="docs-object-method">&nbsp;</a> 
```python
extract_styles(attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L398)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L398?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.get_class_map" class="docs-object-method">&nbsp;</a> 
```python
get_class_map(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L1039)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L1039?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(etree: xml.etree.ElementTree.Element, strip=True, converter=None, **extra_attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L1073)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L1073?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.parse" class="docs-object-method">&nbsp;</a> 
```python
parse(str, strict=True, strip=True, converter=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/HTML.py#L1110)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L1110?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Jupyter/JHTML/HTML/HTML.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Jupyter/JHTML/HTML/HTML.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Jupyter/JHTML/HTML/HTML.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Jupyter/JHTML/HTML/HTML.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/HTML.py#L348?message=Update%20Docs)