## <a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller">NDarrayMarshaller</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L392)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L392?message=Update%20Docs)]
</div>

Support class for `HDF5Serializer` and other
NumPy-friendly interfaces that marshalls data
to/from NumPy arrays

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
atomic_types: tuple
```
<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, base_serializer=None, allow_pickle=True, psuedopickler=None, allow_records=False, all_dicts=False, converters=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L399)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L399?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.get_default_converters" class="docs-object-method">&nbsp;</a> 
```python
get_default_converters(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L421)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L421?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.converter_dispatch" class="docs-object-method">&nbsp;</a> 
```python
@property
converter_dispatch(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NDarrayMarshaller.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, data, allow_pickle=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L559)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L559?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L603)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L603?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L653)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L653?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/NDarrayMarshaller.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L392?message=Update%20Docs)