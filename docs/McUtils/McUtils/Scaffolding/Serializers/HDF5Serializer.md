## <a id="McUtils.McUtils.Scaffolding.Serializers.HDF5Serializer">HDF5Serializer</a>
Defines a serializer that can prep/dump python data to HDF5.
To minimize complexity, we always use NumPy as an interface layer.
This restricts what we can serialize, but generally in insignificant ways.

### Properties and Methods
```python
atomic_types: tuple
converter_dispatch: OrderedDict
```
<a id="McUtils.McUtils.Scaffolding.Serializers.HDF5Serializer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.HDF5Serializer.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, data): 
```
Converts data into format that can be serialized easily
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Serializers.HDF5Serializer.serialize" class="docs-object-method">&nbsp;</a>
```python
serialize(self, file, data, **kwargs): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.HDF5Serializer.deconvert" class="docs-object-method">&nbsp;</a>
```python
deconvert(self, data): 
```
Converts an HDF5 Dataset into a NumPy array or Group into a dict
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Serializers.HDF5Serializer.deserialize" class="docs-object-method">&nbsp;</a>
```python
deserialize(self, file, key=None, **kwargs): 
```

### Examples


