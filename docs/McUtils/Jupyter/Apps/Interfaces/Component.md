## <a id="McUtils.Jupyter.Apps.Interfaces.Component">Component</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L50)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L50?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L55)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L55?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L68)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L68?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.get_child" class="docs-object-method">&nbsp;</a> 
```python
get_child(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L70)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L70?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L74)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L74?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.set_attr" class="docs-object-method">&nbsp;</a> 
```python
set_attr(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L80)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L80?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.update_widget_attr" class="docs-object-method">&nbsp;</a> 
```python
update_widget_attr(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L82)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L82?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.set_child" class="docs-object-method">&nbsp;</a> 
```python
set_child(self, which, new): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L84)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L84?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.update_widget_child" class="docs-object-method">&nbsp;</a> 
```python
update_widget_child(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L88)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L88?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.__setitem__" class="docs-object-method">&nbsp;</a> 
```python
__setitem__(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L90)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L90?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_attr" class="docs-object-method">&nbsp;</a> 
```python
del_attr(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L100)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L100?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_widget_attr" class="docs-object-method">&nbsp;</a> 
```python
del_widget_attr(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L102)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L102?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_child" class="docs-object-method">&nbsp;</a> 
```python
del_child(self, which): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L104)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L104?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.del_widget_child" class="docs-object-method">&nbsp;</a> 
```python
del_widget_child(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L108)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L108?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.__delitem__" class="docs-object-method">&nbsp;</a> 
```python
__delitem__(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L111)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L111?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.to_jhtml" class="docs-object-method">&nbsp;</a> 
```python
to_jhtml(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L123)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L123?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.to_widget" class="docs-object-method">&nbsp;</a> 
```python
to_widget(self, parent=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L126)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L126?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.mutate" class="docs-object-method">&nbsp;</a> 
```python
mutate(self, fn): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L134)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L134?message=Update%20Docs)]
</div>

<a id="McUtils.Jupyter.Apps.Interfaces.Component.invalidate_cache" class="docs-object-method">&nbsp;</a> 
```python
invalidate_cache(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Jupyter/Apps/Interfaces.py#L137)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L137?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Jupyter/Apps/Interfaces/Component.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Jupyter/Apps/Interfaces/Component.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Jupyter/Apps/Interfaces/Component.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Jupyter/Apps/Interfaces/Component.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Jupyter/Apps/Interfaces.py#L50?message=Update%20Docs)