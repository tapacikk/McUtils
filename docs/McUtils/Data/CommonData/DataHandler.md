## <a id="McUtils.Data.CommonData.DataHandler">DataHandler</a>
Defines a general data loader class that we can use for `AtomData` and any other data classes we might find useful.

### Properties and Methods
<a id="McUtils.Data.CommonData.DataHandler.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, data_name, data_key=None, source_key=None, data_dir=None, data_pkg=None, alternate_keys=None, getter=None, record_type=None): 
```

- `data_name`: `str`
    >the name of the dataset
- `data_key`: `str | None`
    >the key in the loaded dictionary to use for the actual data (`"data"` by default)
- `source_key`: `str | None`
    >the key in the loaded dictionary for the original data source (`"source"` by default)
- `data_dir`: `str | None`
    >the main directory data will be loaded from (`.` by default)
- `data_pkg`: `str | None`
    >the python package to load (`TheRealMcCoy` by default)
- `alternate_keys`: `Iterable[str] | None`
    >alternate keys that can be used to index into the dataset which can will be populated at runtime
- `getter`: `callable | None`
    >a function to use to resolve a key
- `record_type`: `type | None`
    >the class to use for holding data (`DataRecord` by default)

<a id="McUtils.Data.CommonData.DataHandler.data_file" class="docs-object-method">&nbsp;</a>
```python
@property
data_file(self): 
```

<a id="McUtils.Data.CommonData.DataHandler.load" class="docs-object-method">&nbsp;</a>
```python
load(self, env=None): 
```
Actually loads the data from `data_file`.
        Currently set up to just use an `import` statement but should
        be reimplemented to use a `Deserializer` from `Scaffolding.Serializers`
- `:returns`: `_`
    >No description...

<a id="McUtils.Data.CommonData.DataHandler.data" class="docs-object-method">&nbsp;</a>
```python
@property
data(self): 
```

<a id="McUtils.Data.CommonData.DataHandler.source" class="docs-object-method">&nbsp;</a>
```python
@property
source(self): 
```

<a id="McUtils.Data.CommonData.DataHandler.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, key): 
```

<a id="McUtils.Data.CommonData.DataHandler.__len__" class="docs-object-method">&nbsp;</a>
```python
__len__(self): 
```

<a id="McUtils.Data.CommonData.DataHandler.__iter__" class="docs-object-method">&nbsp;</a>
```python
__iter__(self): 
```

<a id="McUtils.Data.CommonData.DataHandler.__getstate__" class="docs-object-method">&nbsp;</a>
```python
__getstate__(self): 
```

<a id="McUtils.Data.CommonData.DataHandler.__setstate__" class="docs-object-method">&nbsp;</a>
```python
__setstate__(self, state): 
```

<a id="McUtils.Data.CommonData.DataHandler.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Data/CommonData/DataHandler.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Data/CommonData/DataHandler.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Data/CommonData/DataHandler.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Data/CommonData/DataHandler.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Data/CommonData.py?message=Update%20Docs)