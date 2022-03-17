## <a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer">DumpCheckpointer</a>
A subclass of `CheckpointerBase` that writes an entire dump to file at once & maintains
a backend cache to update it cleanly

### Properties and Methods
<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, file, cache=None, open_kwargs=None, allowed_keys=None, omitted_keys=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L214)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L214?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.load_cache" class="docs-object-method">&nbsp;</a> 
```python
load_cache(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L223)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L223?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.__enter__" class="docs-object-method">&nbsp;</a> 
```python
__enter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L226)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L226?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.__exit__" class="docs-object-method">&nbsp;</a> 
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L229)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L229?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.dump" class="docs-object-method">&nbsp;</a> 
```python
dump(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L234)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L234?message=Update%20Docs)]
</div>

Writes the entire data structure
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L242)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L242?message=Update%20Docs)]
</div>

Converts the cache to an exportable form if needed
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.open_checkpoint_file" class="docs-object-method">&nbsp;</a> 
```python
open_checkpoint_file(self, chk): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L249)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L249?message=Update%20Docs)]
</div>

Opens the passed `checkpoint_file` (if not already open)
- `chk`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.close_checkpoint_file" class="docs-object-method">&nbsp;</a> 
```python
close_checkpoint_file(self, stream): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L260)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L260?message=Update%20Docs)]
</div>

Closes the opened checkpointing stream
- `stream`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.save_parameter" class="docs-object-method">&nbsp;</a> 
```python
save_parameter(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L270)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L270?message=Update%20Docs)]
</div>

Saves a parameter to the checkpoint file
- `key`: `Any`
    >No description...
- `value`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.load_parameter" class="docs-object-method">&nbsp;</a> 
```python
load_parameter(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L281)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L281?message=Update%20Docs)]
</div>

Loads a parameter from the checkpoint file
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.DumpCheckpointer.keys" class="docs-object-method">&nbsp;</a> 
```python
keys(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L291)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L291?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Scaffolding/Checkpointing/DumpCheckpointer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Scaffolding/Checkpointing/DumpCheckpointer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Scaffolding/Checkpointing/DumpCheckpointer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Scaffolding/Checkpointing/DumpCheckpointer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Checkpointing.py?message=Update%20Docs)