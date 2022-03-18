## <a id="McUtils.Scaffolding.Configurations.ParameterManager">ParameterManager</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L187)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L187?message=Update%20Docs)]
</div>

Provides a helpful manager for those cases where
there are way too many options and we need to filter
them across subclasses and things

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Scaffolding.Configurations.ParameterManager.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *d, **ops): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L194)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L194?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.__getattr__" class="docs-object-method">&nbsp;</a> 
```python
__getattr__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L203)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L203?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.__setattr__" class="docs-object-method">&nbsp;</a> 
```python
__setattr__(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L205)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L205?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.__delattr__" class="docs-object-method">&nbsp;</a> 
```python
__delattr__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L210)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L210?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.__hasattr__" class="docs-object-method">&nbsp;</a> 
```python
__hasattr__(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L212)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L212?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.update" class="docs-object-method">&nbsp;</a> 
```python
update(self, **ops): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L214)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L214?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.keys" class="docs-object-method">&nbsp;</a> 
```python
keys(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L217)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L217?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.items" class="docs-object-method">&nbsp;</a> 
```python
items(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L219)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L219?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.save" class="docs-object-method">&nbsp;</a> 
```python
save(self, file, mode=None, attribute=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L222)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L222?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.load" class="docs-object-method">&nbsp;</a> 
```python
load(file, mode=None, attribute=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L224)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L224?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.get_props" class="docs-object-method">&nbsp;</a> 
```python
get_props(self, obj): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L228)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L228?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.bind" class="docs-object-method">&nbsp;</a> 
```python
bind(self, obj, props=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L238)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L238?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.filter" class="docs-object-method">&nbsp;</a> 
```python
filter(self, obj, props=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L243)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L243?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, file, mode=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L253)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L253?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.ParameterManager.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(file, mode=None, attribute=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Configurations.py#L256)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L256?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Configurations/ParameterManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Configurations/ParameterManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Configurations/ParameterManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Configurations/ParameterManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Configurations.py#L187?message=Update%20Docs)