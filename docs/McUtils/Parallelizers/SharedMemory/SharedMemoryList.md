## <a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList">SharedMemoryList</a>
Implements a shared dict that uses
a managed dict to synchronize array metainfo
across processes

### Properties and Methods
<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, *seq, sync_list=None, manager=None, marshaller=None, allocator=None, parallelizer=None): 
```

- `marshaller`: `Any`
    >No description...
- `sync_dict`: `Any`
    >No description...
- `allocator`: `Any`
    >No description...
- `parallelizer`: `Any`
    >No description...

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__getstate__" class="docs-object-method">&nbsp;</a>
```python
__getstate__(self): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__contains__" class="docs-object-method">&nbsp;</a>
```python
__contains__(self, item): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__iter__" class="docs-object-method">&nbsp;</a>
```python
__iter__(self): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__len__" class="docs-object-method">&nbsp;</a>
```python
__len__(self): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.unshare" class="docs-object-method">&nbsp;</a>
```python
unshare(self): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.pop" class="docs-object-method">&nbsp;</a>
```python
pop(self, k=0): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.insert" class="docs-object-method">&nbsp;</a>
```python
insert(self, k, v): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.append" class="docs-object-method">&nbsp;</a>
```python
append(self, v): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.extend" class="docs-object-method">&nbsp;</a>
```python
extend(self, v): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/SharedMemory.py?message=Update%20Docs)