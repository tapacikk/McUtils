## <a id="McUtils.Scaffolding.Serializers.NumPySerializer">NumPySerializer</a>
A serializer that implements NPZ dumps

### Properties and Methods
```python
default_extension: str
atomic_types: tuple
converter_dispatch: OrderedDict
dict_key_sep: str
```
<a id="McUtils.Scaffolding.Serializers.NumPySerializer.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, data): 
```

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.deconvert" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.serialize" class="docs-object-method">&nbsp;</a>
```python
serialize(self, file, data, **kwargs): 
```

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.deserialize" class="docs-object-method">&nbsp;</a>
```python
deserialize(self, file, key=None, **kwargs): 
```

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Serializers/NumPySerializer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Serializers/NumPySerializer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Serializers/NumPySerializer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Serializers/NumPySerializer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Serializers.py?message=Update%20Docs)