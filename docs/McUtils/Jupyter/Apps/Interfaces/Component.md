## <a id="McUtils.Jupyter.Apps.Interfaces.Component">Component</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L51)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L51?message=Update%20Docs)]
</div>

Provides an abstract base class for an interface element
to allow for the easy construction of interesting interfaces

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Jupyter.Apps.Interfaces.Component.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, dynamic=True, debug_pane=None, **attrs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L56)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L56?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.attrs" class="docs-object-method">&nbsp;</a> 
```python
@property
attrs(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.get_attr" class="docs-object-method">&nbsp;</a> 
```python
get_attr(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L69)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L69?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.get_child" class="docs-object-method">&nbsp;</a> 
```python
get_child(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L71)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L71?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L75)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L75?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.set_attr" class="docs-object-method">&nbsp;</a> 
```python
set_attr(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L81)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L81?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.update_widget_attr" class="docs-object-method">&nbsp;</a> 
```python
update_widget_attr(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L83)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L83?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.set_child" class="docs-object-method">&nbsp;</a> 
```python
set_child(self, which, new): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L85)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L85?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.update_widget_child" class="docs-object-method">&nbsp;</a> 
```python
update_widget_child(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L89)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L89?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.__setitem__" class="docs-object-method">&nbsp;</a> 
```python
__setitem__(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L91)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L91?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_attr" class="docs-object-method">&nbsp;</a> 
```python
del_attr(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L101)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L101?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_widget_attr" class="docs-object-method">&nbsp;</a> 
```python
del_widget_attr(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L103)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L103?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_child" class="docs-object-method">&nbsp;</a> 
```python
del_child(self, which): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L105)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L105?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_widget_child" class="docs-object-method">&nbsp;</a> 
```python
del_widget_child(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L109)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L109?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.__delitem__" class="docs-object-method">&nbsp;</a> 
```python
__delitem__(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L112)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L112?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.to_jhtml" class="docs-object-method">&nbsp;</a> 
```python
to_jhtml(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L124)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L124?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.to_widget" class="docs-object-method">&nbsp;</a> 
```python
to_widget(self, parent=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L127)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L127?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.mutate" class="docs-object-method">&nbsp;</a> 
```python
mutate(self, fn): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L135)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L135?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.invalidate_cache" class="docs-object-method">&nbsp;</a> 
```python
invalidate_cache(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L138)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L138?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Jupyter/Apps/Interfaces/Component.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Jupyter/Apps/Interfaces/Component.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Jupyter/Apps/Interfaces/Component.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Jupyter/Apps/Interfaces/Component.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L51?message=Update%20Docs)