## <a id="McUtils.Scaffolding.Checkpointing.NullCheckpointer">NullCheckpointer</a>
A checkpointer that doesn't actually do anything, but which is provided
so that programs can turn off checkpointing without changing their layout

### Properties and Methods
<a id="McUtils.Scaffolding.Checkpointing.NullCheckpointer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, checkpoint_file=None): 
```

<a id="McUtils.Scaffolding.Checkpointing.NullCheckpointer.open_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
open_checkpoint_file(self, chk): 
```
Opens the passed `checkpoint_file` (if not already open)
- `chk`: `str | file-like`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.NullCheckpointer.close_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
close_checkpoint_file(self, stream): 
```
Opens the passed `checkpoint_file` (if not already open)
- `chk`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.NullCheckpointer.save_parameter" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Scaffolding.Checkpointing.NullCheckpointer.load_parameter" class="docs-object-method">&nbsp;</a>
```python
load_parameter(self, key): 
```
Loads a parameter from the checkpoint file
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.NullCheckpointer.keys" class="docs-object-method">&nbsp;</a>
```python
keys(self): 
```

### Examples


