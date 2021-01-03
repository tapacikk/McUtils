## <a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase">CheckpointerBase</a>
General purpose base class that allows checkpointing to be done easily and cleanly.
Intended to be a passable object that allows code to checkpoint easily.

### Properties and Methods
<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, checkpoint_file): 
```

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.__enter__" class="docs-object-method">&nbsp;</a>
```python
__enter__(self): 
```

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.__exit__" class="docs-object-method">&nbsp;</a>
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.is_open" class="docs-object-method">&nbsp;</a>
```python
@property
is_open(self): 
```

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.stream" class="docs-object-method">&nbsp;</a>
```python
@property
stream(self): 
```

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.open_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
open_checkpoint_file(self, chk): 
```
Opens the passed `checkpoint_file` (if not already open)
- `chk`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.close_checkpoint_file" class="docs-object-method">&nbsp;</a>
```python
close_checkpoint_file(self, stream): 
```
Closes the opened checkpointing stream
- `stream`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.save_parameter" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.load_parameter" class="docs-object-method">&nbsp;</a>
```python
load_parameter(self, key): 
```
Loads a parameter from the checkpoint file
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, item): 
```

<a id="McUtils.Scaffolding.Checkpointing.CheckpointerBase.__setitem__" class="docs-object-method">&nbsp;</a>
```python
__setitem__(self, key, value): 
```

### Examples


