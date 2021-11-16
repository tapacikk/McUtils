## <a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller">NDarrayMarshaller</a>
Support class for `HDF5Serializer` and other
NumPy-friendly interfaces that marshalls data
to/from NumPy arrays

### Properties and Methods
```python
atomic_types: tuple
default_converter_dispatch: OrderedDict
```
<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, base_serializer, allow_pickle=True, psuedopickler=None, allow_records=False, all_dicts=False, converters=None): 
```

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, data, allow_pickle=None): 
```
Recursively loop through, test data, make sure HDF5 compatible
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.deconvert" class="docs-object-method">&nbsp;</a>
```python
deconvert(self, data): 
```
Reverses the conversion process
        used to marshall the data
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, data, allow_pickle=None): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Serializers.py?message=Update%20Docs)