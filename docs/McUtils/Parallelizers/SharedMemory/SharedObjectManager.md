## <a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager">SharedObjectManager</a>
Provides a high-level interface to create a manager
that supports shared memory objects through the multiprocessing
interface
Only supports data that can be marshalled into a NumPy array.

### Properties and Methods
```python
primitive_types: tuple
```
<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, obj, base_dict=None, parallelizer=None): 
```

- `mem_manager`: `Any`
    >a memory manager like `multiprocessing.SharedMemoryManager`
- `obj`: `Any`
    >the object whose attributes should be given by shared memory objects
- `base_dict`: `SharedMemoryDict`
    >the dict that stores the shared arrays (can also be shared)

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.is_primitive" class="docs-object-method">&nbsp;</a>
```python
is_primitive(val): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.save_attr" class="docs-object-method">&nbsp;</a>
```python
save_attr(self, attr): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.del_attr" class="docs-object-method">&nbsp;</a>
```python
del_attr(self, attr): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.load_attr" class="docs-object-method">&nbsp;</a>
```python
load_attr(self, attr): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.get_saved_keys" class="docs-object-method">&nbsp;</a>
```python
get_saved_keys(self, obj): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.save_keys" class="docs-object-method">&nbsp;</a>
```python
save_keys(self, keys=None): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.share" class="docs-object-method">&nbsp;</a>
```python
share(self, keys=None): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.load_keys" class="docs-object-method">&nbsp;</a>
```python
load_keys(self, keys=None): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.unshare" class="docs-object-method">&nbsp;</a>
```python
unshare(self, keys=None): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.__del__" class="docs-object-method">&nbsp;</a>
```python
__del__(self): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.list" class="docs-object-method">&nbsp;</a>
```python
list(self, *l): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.dict" class="docs-object-method">&nbsp;</a>
```python
dict(self, *d): 
```

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.array" class="docs-object-method">&nbsp;</a>
```python
array(self, a): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/SharedMemory.py?message=Update%20Docs)