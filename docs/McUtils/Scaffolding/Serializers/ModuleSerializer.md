## <a id="McUtils.Scaffolding.Serializers.ModuleSerializer">ModuleSerializer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L1031)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L1031?message=Update%20Docs)]
</div>

A somewhat hacky serializer that supports module-based serialization.
Writes all module parameters to a dict with a given attribute.
Serialization doesn't support loading arbitrary python code, but deserialization does.
Use at your own risk.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_extension: str
default_loader: NoneType
default_attr: str
```
<a id="McUtils.Scaffolding.Serializers.ModuleSerializer.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, attr=None, loader=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L1043)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L1043?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.ModuleSerializer.loader" class="docs-object-method">&nbsp;</a> 
```python
@property
loader(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.ModuleSerializer.attr" class="docs-object-method">&nbsp;</a> 
```python
@property
attr(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.ModuleSerializer.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L1066)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L1066?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.ModuleSerializer.deconvert" class="docs-object-method">&nbsp;</a> 
```python
deconvert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L1068)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L1068?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.ModuleSerializer.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, file, data, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L1070)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L1070?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.ModuleSerializer.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(self, file, key=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L1082)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L1082?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Serializers/ModuleSerializer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Serializers/ModuleSerializer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Serializers/ModuleSerializer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/ModuleSerializer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L1031?message=Update%20Docs)