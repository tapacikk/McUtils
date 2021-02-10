## <a id="McUtils.Scaffolding.Checkpointing.HDF5Checkpointer">HDF5Checkpointer</a>
A checkpointer that uses an HDF5 file as a backend.
Doesn't maintain a secondary `dict`, because HDF5 is an updatable format.

### Properties and Methods
<a id="McUtils.Scaffolding.Checkpointing.HDF5Checkpointer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, checkpoint_file, serializer=None): 
```

<a id="McUtils.Scaffolding.Checkpointing.HDF5Checkpointer.open_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
open_checkpoint_file(self, chk): 
```
Opens the passed `checkpoint_file` (if not already open)
- `chk`: `str | file-like`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.HDF5Checkpointer.close_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
close_checkpoint_file(self, stream): 
```
Opens the passed `checkpoint_file` (if not already open)
- `chk`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.HDF5Checkpointer.save_parameter" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Scaffolding.Checkpointing.HDF5Checkpointer.load_parameter" class="docs-object-method">&nbsp;</a>
```python
load_parameter(self, key): 
```
Loads a parameter from the checkpoint file
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.HDF5Checkpointer.keys" class="docs-object-method">&nbsp;</a>
```python
keys(self): 
```

### Examples


