## <a id="McUtils.McUtils.Data.CommonData.DataRecord">DataRecord</a>
Represents an individual record that might be accessed from a `DataHandler`.
Implements _most_ of the `dict` interface, but to make things a bit easier when
pickling is not implemented as a proper subclass of `dict`.

### Properties and Methods
<a id="McUtils.McUtils.Data.CommonData.DataRecord.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, data_handler, key, records): 
```

<a id="McUtils.McUtils.Data.CommonData.DataRecord.keys" class="docs-object-method">&nbsp;</a>
```python
keys(self): 
```

<a id="McUtils.McUtils.Data.CommonData.DataRecord.values" class="docs-object-method">&nbsp;</a>
```python
values(self): 
```

<a id="McUtils.McUtils.Data.CommonData.DataRecord.items" class="docs-object-method">&nbsp;</a>
```python
items(self): 
```

<a id="McUtils.McUtils.Data.CommonData.DataRecord.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, item): 
```

<a id="McUtils.McUtils.Data.CommonData.DataRecord.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

<a id="McUtils.McUtils.Data.CommonData.DataRecord.__getstate__" class="docs-object-method">&nbsp;</a>
```python
__getstate__(self): 
```

<a id="McUtils.McUtils.Data.CommonData.DataRecord.__setstate__" class="docs-object-method">&nbsp;</a>
```python
__setstate__(self, state): 
```

### Examples
