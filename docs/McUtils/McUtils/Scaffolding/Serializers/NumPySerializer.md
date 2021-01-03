## <a id="McUtils.McUtils.Scaffolding.Serializers.NumPySerializer">NumPySerializer</a>
A serializer that makes implements NPZ dumps

### Properties and Methods
```python
atomic_types: tuple
converter_dispatch: OrderedDict
dict_key_sep: str
```
<a id="McUtils.McUtils.Scaffolding.Serializers.NumPySerializer.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, data): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.NumPySerializer.deconvert" class="docs-object-method">&nbsp;</a>
```python
deconvert(self, data, sep=None): 
```
Unflattens nested dictionary structures so that the original data
        can be recovered
- `data`: `Any`
    >No description...
- `sep`: `str | None`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Serializers.NumPySerializer.serialize" class="docs-object-method">&nbsp;</a>
```python
serialize(self, file, data, **kwargs): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.NumPySerializer.deserialize" class="docs-object-method">&nbsp;</a>
```python
deserialize(self, file, key=None, **kwargs): 
```

### Examples


