## <a id="McUtils.Scaffolding.Serializers.BaseSerializer">BaseSerializer</a>
Serializer base class to define the interface

### Properties and Methods
```python
default_extension: str
```
<a id="McUtils.Scaffolding.Serializers.BaseSerializer.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, data): 
```
Converts data into a serializable format
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.BaseSerializer.deconvert" class="docs-object-method">&nbsp;</a>
```python
deconvert(self, data): 
```
Converts data from a serialized format into a python format
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.BaseSerializer.serialize" class="docs-object-method">&nbsp;</a>
```python
serialize(self, file, data, **kwargs): 
```
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
Loads data from a file
- `file`: `Any`
    >No description...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Serializers/BaseSerializer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Serializers/BaseSerializer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Serializers/BaseSerializer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Serializers/BaseSerializer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Serializers.py?message=Update%20Docs)