## <a id="McUtils.McUtils.Scaffolding.Serializers.BaseSerializer">BaseSerializer</a>
Serializer base class to define the interface

### Properties and Methods
<a id="McUtils.McUtils.Scaffolding.Serializers.BaseSerializer.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, data): 
```
Converts data into a serializable format
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Serializers.BaseSerializer.deconvert" class="docs-object-method">&nbsp;</a>
```python
deconvert(self, data): 
```
Converts data from a serialized format into a python format
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Serializers.BaseSerializer.serialize" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.McUtils.Scaffolding.Serializers.BaseSerializer.deserialize" class="docs-object-method">&nbsp;</a>
```python
deserialize(self, file, **kwargs): 
```
Loads data from a file
- `file`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples
