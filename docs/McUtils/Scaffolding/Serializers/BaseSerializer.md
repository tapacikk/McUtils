## <a id="McUtils.Scaffolding.Serializers.BaseSerializer">BaseSerializer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L246)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L246?message=Update%20Docs)]
</div>

Serializer base class to define the interface

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_extension: str
```
<a id="McUtils.Scaffolding.Serializers.BaseSerializer.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L253)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L253?message=Update%20Docs)]
</div>

Converts data into a serializable format
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.BaseSerializer.deconvert" class="docs-object-method">&nbsp;</a> 
```python
deconvert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L263)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L263?message=Update%20Docs)]
</div>

Converts data from a serialized format into a python format
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.BaseSerializer.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, file, data, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L273)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L273?message=Update%20Docs)]
</div>

Writes the data
- `file`: `Any`
    >No description...
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.BaseSerializer.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(self, file, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L285)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L285?message=Update%20Docs)]
</div>

Loads data from a file
- `file`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Serializers/BaseSerializer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Serializers/BaseSerializer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Serializers/BaseSerializer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/BaseSerializer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L246?message=Update%20Docs)