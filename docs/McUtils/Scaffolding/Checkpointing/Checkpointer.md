## <a id="McUtils.Scaffolding.Checkpointing.Checkpointer">Checkpointer</a>
General purpose base class that allows checkpointing to be done easily and cleanly.
Intended to be a passable object that allows code to checkpoint easily.

### Properties and Methods
```python
default_extension: str
```
<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, checkpoint_file, allowed_keys=None, omitted_keys=None): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.extension_map" class="docs-object-method">&nbsp;</a>
```python
extension_map(): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.build_canonical" class="docs-object-method">&nbsp;</a>
```python
build_canonical(checkpoint): 
```
Dispatches over types of objects to make a canonical checkpointer
        from the supplied data
- `checkpoint`: `None | str | Checkpoint | file | dict`
    >provides
- `:returns`: `Checkpointer`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.from_file" class="docs-object-method">&nbsp;</a>
```python
from_file(file, **opts): 
```
Dispatch function to load from the appropriate file
- `file`: `str | File`
    >No description...
- `opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__enter__" class="docs-object-method">&nbsp;</a>
```python
__enter__(self): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__exit__" class="docs-object-method">&nbsp;</a>
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.is_open" class="docs-object-method">&nbsp;</a>
```python
@property
is_open(self): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.stream" class="docs-object-method">&nbsp;</a>
```python
@property
stream(self): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.open_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
open_checkpoint_file(self, chk): 
```
Opens the passed `checkpoint_file` (if not already open)
- `chk`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.close_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
close_checkpoint_file(self, stream): 
```
Closes the opened checkpointing stream
- `stream`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.save_parameter" class="docs-object-method">&nbsp;</a>
```python
save_parameter(self, key, value): 
```
Saves a parameter to the checkpoint file
- `key`: `Any`
    >No description...
- `value`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.load_parameter" class="docs-object-method">&nbsp;</a>
```python
load_parameter(self, key): 
```
Loads a parameter from the checkpoint file
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.check_allowed_key" class="docs-object-method">&nbsp;</a>
```python
check_allowed_key(self, item): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, item): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__setitem__" class="docs-object-method">&nbsp;</a>
```python
__setitem__(self, key, value): 
```

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.keys" class="docs-object-method">&nbsp;</a>
```python
keys(self): 
```
Returns the keys of currently checkpointed
        objects
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Checkpointing/Checkpointer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Checkpointing/Checkpointer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Checkpointing/Checkpointer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Checkpointing/Checkpointer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Checkpointing.py?message=Update%20Docs)