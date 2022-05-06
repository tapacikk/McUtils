## <a id="McUtils.Jupyter.JHTML.JHTML.JHTML">JHTML</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L14)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L14?message=Update%20Docs)]
</div>

Provides dispatchers to either pure HTML components or Widget components based on whether interactivity
is required or not

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
callbacks: dict
widgets: dict
Bootstrap: type
Styled: type
Compound: type
```
<a id="McUtils.Jupyter.JHTML.HTML.HTML.manage_class" class="docs-object-method">&nbsp;</a> 
```python
manage_cls(cls): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L358)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L358?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.manage_styles" class="docs-object-method">&nbsp;</a> 
```python
manage_style(styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L372)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L372?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.extract_styles" class="docs-object-method">&nbsp;</a> 
```python
extract_styles(attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L392)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L392?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.HTML.HTML.manage_attrs" class="docs-object-method">&nbsp;</a> 
```python
manage_attrs(attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L384)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L384?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.load" class="docs-object-method">&nbsp;</a> 
```python
load(overwrite=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L23)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L23?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, context=None, include_bootstrap=False, expose_classes=True, output_pane=True, callbacks=None, widgets=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L33)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L33?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.insert_vars" class="docs-object-method">&nbsp;</a> 
```python
insert_vars(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L58)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L58?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.wrap_callbacks" class="docs-object-method">&nbsp;</a> 
```python
wrap_callbacks(self, c): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L73)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L73?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.__enter__" class="docs-object-method">&nbsp;</a> 
```python
__enter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L88)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L88?message=Update%20Docs)]
</div>

To make writing HTML interactively a bit nicer
- `:returns`: `_`
    >No description...

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.out" class="docs-object-method">&nbsp;</a> 
```python
@property
out(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.prune_vars" class="docs-object-method">&nbsp;</a> 
```python
prune_vars(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L120)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L120?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.__exit__" class="docs-object-method">&nbsp;</a> 
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L129)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L129?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.parse_handlers" class="docs-object-method">&nbsp;</a> 
```python
parse_handlers(handler_string): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L142)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L142?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.parse_widget" class="docs-object-method">&nbsp;</a> 
```python
parse_widget(uuid): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L158)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L158?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(etree, strip=True, converter=None, **extra_attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L166)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L166?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.parse" class="docs-object-method">&nbsp;</a> 
```python
parse(src, event_handlers=None, dynamic=None, track_value=None, strict=True, **attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L195)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L195?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Abbr" class="docs-object-method">&nbsp;</a> 
```python
Abbr(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Address" class="docs-object-method">&nbsp;</a> 
```python
Address(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Anchor" class="docs-object-method">&nbsp;</a> 
```python
Anchor(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Anchor" class="docs-object-method">&nbsp;</a> 
```python
A(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Area" class="docs-object-method">&nbsp;</a> 
```python
Area(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Article" class="docs-object-method">&nbsp;</a> 
```python
Article(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Aside" class="docs-object-method">&nbsp;</a> 
```python
Aside(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Audio" class="docs-object-method">&nbsp;</a> 
```python
Audio(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.B" class="docs-object-method">&nbsp;</a> 
```python
B(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Base" class="docs-object-method">&nbsp;</a> 
```python
Base(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Bdi" class="docs-object-method">&nbsp;</a> 
```python
Bdi(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Bdo" class="docs-object-method">&nbsp;</a> 
```python
Bdo(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Blockquote" class="docs-object-method">&nbsp;</a> 
```python
Blockquote(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Body" class="docs-object-method">&nbsp;</a> 
```python
Body(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Bold" class="docs-object-method">&nbsp;</a> 
```python
Bold(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Br" class="docs-object-method">&nbsp;</a> 
```python
Br(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Button" class="docs-object-method">&nbsp;</a> 
```python
Button(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Canvas" class="docs-object-method">&nbsp;</a> 
```python
Canvas(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Caption" class="docs-object-method">&nbsp;</a> 
```python
Caption(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Cite" class="docs-object-method">&nbsp;</a> 
```python
Cite(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Code" class="docs-object-method">&nbsp;</a> 
```python
Code(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Col" class="docs-object-method">&nbsp;</a> 
```python
Col(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Colgroup" class="docs-object-method">&nbsp;</a> 
```python
Colgroup(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Data" class="docs-object-method">&nbsp;</a> 
```python
Data(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Datalist" class="docs-object-method">&nbsp;</a> 
```python
Datalist(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Dd" class="docs-object-method">&nbsp;</a> 
```python
Dd(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Del" class="docs-object-method">&nbsp;</a> 
```python
Del(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Details" class="docs-object-method">&nbsp;</a> 
```python
Details(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Dfn" class="docs-object-method">&nbsp;</a> 
```python
Dfn(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Dialog" class="docs-object-method">&nbsp;</a> 
```python
Dialog(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Div" class="docs-object-method">&nbsp;</a> 
```python
Div(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Dl" class="docs-object-method">&nbsp;</a> 
```python
Dl(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Dt" class="docs-object-method">&nbsp;</a> 
```python
Dt(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Em" class="docs-object-method">&nbsp;</a> 
```python
Em(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Embed" class="docs-object-method">&nbsp;</a> 
```python
Embed(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Fieldset" class="docs-object-method">&nbsp;</a> 
```python
Fieldset(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Figcaption" class="docs-object-method">&nbsp;</a> 
```python
Figcaption(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Figure" class="docs-object-method">&nbsp;</a> 
```python
Figure(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Footer" class="docs-object-method">&nbsp;</a> 
```python
Footer(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Form" class="docs-object-method">&nbsp;</a> 
```python
Form(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Head" class="docs-object-method">&nbsp;</a> 
```python
Head(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Header" class="docs-object-method">&nbsp;</a> 
```python
Header(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Heading" class="docs-object-method">&nbsp;</a> 
```python
Heading(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Hr" class="docs-object-method">&nbsp;</a> 
```python
Hr(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Html" class="docs-object-method">&nbsp;</a> 
```python
Html(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Iframe" class="docs-object-method">&nbsp;</a> 
```python
Iframe(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Image" class="docs-object-method">&nbsp;</a> 
```python
Image(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Img" class="docs-object-method">&nbsp;</a> 
```python
Img(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Input" class="docs-object-method">&nbsp;</a> 
```python
Input(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Ins" class="docs-object-method">&nbsp;</a> 
```python
Ins(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Italic" class="docs-object-method">&nbsp;</a> 
```python
Italic(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Italic" class="docs-object-method">&nbsp;</a> 
```python
I(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Kbd" class="docs-object-method">&nbsp;</a> 
```python
Kbd(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Label" class="docs-object-method">&nbsp;</a> 
```python
Label(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Legend" class="docs-object-method">&nbsp;</a> 
```python
Legend(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Link" class="docs-object-method">&nbsp;</a> 
```python
Link(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.List" class="docs-object-method">&nbsp;</a> 
```python
List(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.List" class="docs-object-method">&nbsp;</a> 
```python
Ul(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.ListItem" class="docs-object-method">&nbsp;</a> 
```python
ListItem(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.ListItem" class="docs-object-method">&nbsp;</a> 
```python
Li(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Main" class="docs-object-method">&nbsp;</a> 
```python
Main(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Map" class="docs-object-method">&nbsp;</a> 
```python
Map(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Mark" class="docs-object-method">&nbsp;</a> 
```python
Mark(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Meta" class="docs-object-method">&nbsp;</a> 
```python
Meta(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Meter" class="docs-object-method">&nbsp;</a> 
```python
Meter(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Nav" class="docs-object-method">&nbsp;</a> 
```python
Nav(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Noscript" class="docs-object-method">&nbsp;</a> 
```python
Noscript(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.NumberedList" class="docs-object-method">&nbsp;</a> 
```python
NumberedList(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.NumberedList" class="docs-object-method">&nbsp;</a> 
```python
Ol(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Object" class="docs-object-method">&nbsp;</a> 
```python
Object(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Optgroup" class="docs-object-method">&nbsp;</a> 
```python
Optgroup(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Option" class="docs-object-method">&nbsp;</a> 
```python
Option(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Output" class="docs-object-method">&nbsp;</a> 
```python
Output(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Param" class="docs-object-method">&nbsp;</a> 
```python
Param(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Picture" class="docs-object-method">&nbsp;</a> 
```python
Picture(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Pre" class="docs-object-method">&nbsp;</a> 
```python
Pre(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Progress" class="docs-object-method">&nbsp;</a> 
```python
Progress(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Q" class="docs-object-method">&nbsp;</a> 
```python
Q(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Rp" class="docs-object-method">&nbsp;</a> 
```python
Rp(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Rt" class="docs-object-method">&nbsp;</a> 
```python
Rt(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Ruby" class="docs-object-method">&nbsp;</a> 
```python
Ruby(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.S" class="docs-object-method">&nbsp;</a> 
```python
S(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Samp" class="docs-object-method">&nbsp;</a> 
```python
Samp(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Script" class="docs-object-method">&nbsp;</a> 
```python
Script(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Section" class="docs-object-method">&nbsp;</a> 
```python
Section(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Select" class="docs-object-method">&nbsp;</a> 
```python
Select(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Small" class="docs-object-method">&nbsp;</a> 
```python
Small(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Source" class="docs-object-method">&nbsp;</a> 
```python
Source(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Span" class="docs-object-method">&nbsp;</a> 
```python
Span(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Strong" class="docs-object-method">&nbsp;</a> 
```python
Strong(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Style" class="docs-object-method">&nbsp;</a> 
```python
Style(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Sub" class="docs-object-method">&nbsp;</a> 
```python
Sub(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.SubHeading" class="docs-object-method">&nbsp;</a> 
```python
SubHeading(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.SubsubHeading" class="docs-object-method">&nbsp;</a> 
```python
SubsubHeading(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.SubsubsubHeading" class="docs-object-method">&nbsp;</a> 
```python
SubsubsubHeading(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.SubHeading5" class="docs-object-method">&nbsp;</a> 
```python
SubHeading5(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.SubHeading6" class="docs-object-method">&nbsp;</a> 
```python
SubHeading6(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Summary" class="docs-object-method">&nbsp;</a> 
```python
Summary(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Sup" class="docs-object-method">&nbsp;</a> 
```python
Sup(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Svg" class="docs-object-method">&nbsp;</a> 
```python
Svg(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Table" class="docs-object-method">&nbsp;</a> 
```python
Table(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableBody" class="docs-object-method">&nbsp;</a> 
```python
TableBody(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableBody" class="docs-object-method">&nbsp;</a> 
```python
Tbody(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableFooter" class="docs-object-method">&nbsp;</a> 
```python
TableFooter(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableFooter" class="docs-object-method">&nbsp;</a> 
```python
Tfoot(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableHeader" class="docs-object-method">&nbsp;</a> 
```python
TableHeader(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableHeader" class="docs-object-method">&nbsp;</a> 
```python
Thead(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableHeading" class="docs-object-method">&nbsp;</a> 
```python
TableHeading(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableHeading" class="docs-object-method">&nbsp;</a> 
```python
Th(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableItem" class="docs-object-method">&nbsp;</a> 
```python
TableItem(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableItem" class="docs-object-method">&nbsp;</a> 
```python
Td(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableRow" class="docs-object-method">&nbsp;</a> 
```python
TableRow(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.TableRow" class="docs-object-method">&nbsp;</a> 
```python
Tr(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Template" class="docs-object-method">&nbsp;</a> 
```python
Template(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Text" class="docs-object-method">&nbsp;</a> 
```python
Text(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Text" class="docs-object-method">&nbsp;</a> 
```python
P(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Textarea" class="docs-object-method">&nbsp;</a> 
```python
Textarea(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Time" class="docs-object-method">&nbsp;</a> 
```python
Time(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Title" class="docs-object-method">&nbsp;</a> 
```python
Title(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Track" class="docs-object-method">&nbsp;</a> 
```python
Track(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.U" class="docs-object-method">&nbsp;</a> 
```python
U(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Var" class="docs-object-method">&nbsp;</a> 
```python
Var(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Video" class="docs-object-method">&nbsp;</a> 
```python
Video(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.Wbr" class="docs-object-method">&nbsp;</a> 
```python
Wbr(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.JHTML.JHTML.JHTML.OutputArea" class="docs-object-method">&nbsp;</a> 
```python
OutputArea(*elements, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/JHTML/JHTML.py#L609)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L609?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Jupyter/JHTML/JHTML/JHTML.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Jupyter/JHTML/JHTML/JHTML.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Jupyter/JHTML/JHTML/JHTML.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Jupyter/JHTML/JHTML/JHTML.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/JHTML/JHTML.py#L14?message=Update%20Docs)