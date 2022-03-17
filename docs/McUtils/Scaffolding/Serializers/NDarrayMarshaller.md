## <a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller">NDarrayMarshaller</a>
Support class for `HDF5Serializer` and other
NumPy-friendly interfaces that marshalls data
to/from NumPy arrays

### Properties and Methods
```python
atomic_types: tuple
```
<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, base_serializer=None, allow_pickle=True, psuedopickler=None, allow_records=False, all_dicts=False, converters=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L399)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L399?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.get_default_converters" class="docs-object-method">&nbsp;</a> 
```python
get_default_converters(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L421)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L421?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.converter_dispatch" class="docs-object-method">&nbsp;</a> 
```python
@property
converter_dispatch(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, data, allow_pickle=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L551)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L551?message=Update%20Docs)]
</div>

Recursively loop through, test data, make sure HDF5 compatible
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.deconvert" class="docs-object-method">&nbsp;</a> 
```python
deconvert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L595)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L595?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L645)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L645?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Serializers.py?message=Update%20Docs)