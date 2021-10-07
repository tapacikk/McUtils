## <a id="McUtils.Data.CommonData.DataRecord">DataRecord</a>
Represents an individual record that might be accessed from a `DataHandler`.
Implements _most_ of the `dict` interface, but to make things a bit easier when
pickling is not implemented as a proper subclass of `dict`.

### Properties and Methods
<a id="McUtils.Data.CommonData.DataRecord.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, data_handler, key, records): 
```

<a id="McUtils.Data.CommonData.DataRecord.keys" class="docs-object-method">&nbsp;</a>
```python
keys(self): 
```

<a id="McUtils.Data.CommonData.DataRecord.values" class="docs-object-method">&nbsp;</a>
```python
values(self): 
```

<a id="McUtils.Data.CommonData.DataRecord.items" class="docs-object-method">&nbsp;</a>
```python
items(self): 
```

<a id="McUtils.Data.CommonData.DataRecord.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, item): 
```

<a id="McUtils.Data.CommonData.DataRecord.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

<a id="McUtils.Data.CommonData.DataRecord.__getstate__" class="docs-object-method">&nbsp;</a>
```python
__getstate__(self): 
```

<a id="McUtils.Data.CommonData.DataRecord.__setstate__" class="docs-object-method">&nbsp;</a>
```python
__setstate__(self, state): 
```

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Data/CommonData/DataRecord.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Data/CommonData/DataRecord.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Data/CommonData/DataRecord.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Data/CommonData/DataRecord.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Data/CommonData.py?message=Update%20Docs)