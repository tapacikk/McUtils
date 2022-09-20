## <a id="McUtils.Jupyter.Apps.Apps.App">App</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps.py#L54)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps.py#L54?message=Update%20Docs)]
</div>

Provides a framework for making Jupyter Apps with the
elements built out in the Interfaces package







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Jupyter.Apps.Apps.App.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, body=None, header=None, footer=None, sidebar=None, toolbar=None, layout='grid', cls='border', output=None, capture_output=True, **attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L59)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L59?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.body" class="docs-object-method">&nbsp;</a> 
```python
@property
body(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L83)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L83?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.header" class="docs-object-method">&nbsp;</a> 
```python
@property
header(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L91)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L91?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.sidebar" class="docs-object-method">&nbsp;</a> 
```python
@property
sidebar(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L99)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L99?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.toolbar" class="docs-object-method">&nbsp;</a> 
```python
@property
toolbar(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L107)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L107?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.footer" class="docs-object-method">&nbsp;</a> 
```python
@property
footer(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L115)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L115?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_navbar_item" class="docs-object-method">&nbsp;</a> 
```python
construct_navbar_item(item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L124)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L124?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_header" class="docs-object-method">&nbsp;</a> 
```python
construct_header(header): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L131)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L131?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_footer" class="docs-object-method">&nbsp;</a> 
```python
construct_footer(footer): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L143)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L143?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_sidebar_item" class="docs-object-method">&nbsp;</a> 
```python
construct_sidebar_item(item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L150)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L150?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_sidebar" class="docs-object-method">&nbsp;</a> 
```python
construct_sidebar(sidebar): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L167)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L167?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_toolbar_item" class="docs-object-method">&nbsp;</a> 
```python
construct_toolbar_item(item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L176)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L176?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_toolbar" class="docs-object-method">&nbsp;</a> 
```python
construct_toolbar(toolbar): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L184)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L184?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.wrap_body" class="docs-object-method">&nbsp;</a> 
```python
wrap_body(self, fn, **styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L199)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L199?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_body_item" class="docs-object-method">&nbsp;</a> 
```python
construct_body_item(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L204)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L204?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_body" class="docs-object-method">&nbsp;</a> 
```python
construct_body(cls, body): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L218)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L218?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.construct_layout" class="docs-object-method">&nbsp;</a> 
```python
construct_layout(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L225)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L225?message=Update%20Docs)]
</div>


<a id="McUtils.Jupyter.Apps.Apps.App.to_jhtml" class="docs-object-method">&nbsp;</a> 
```python
to_jhtml(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Apps/App.py#L291)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps/App.py#L291?message=Update%20Docs)]
</div>
 </div>
</div>












---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Jupyter/Apps/Apps/App.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Jupyter/Apps/Apps/App.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Jupyter/Apps/Apps/App.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Jupyter/Apps/Apps/App.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Apps.py#L54?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>