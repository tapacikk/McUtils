## <a id="McUtils.Misc.InteractiveTools.ModuleReloader">ModuleReloader</a>
Reloads a module & recursively descends its 'all' tree
to make sure that all submodules are also reloaded

### Properties and Methods
```python
blacklist_keys: list
```
<a id="McUtils.Misc.InteractiveTools.ModuleReloader.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, modspec): 
```

- `modspec`: `str | types.ModuleType`
    >No description...

<a id="McUtils.Misc.InteractiveTools.ModuleReloader.get_parents" class="docs-object-method">&nbsp;</a>
```python
get_parents(self): 
```
Returns module parents
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.InteractiveTools.ModuleReloader.get_members" class="docs-object-method">&nbsp;</a>
```python
get_members(self): 
```
Returns module members
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.InteractiveTools.ModuleReloader.reload" class="docs-object-method">&nbsp;</a>
```python
reload(self, reloaded=None, blacklist=None, reload_parents=True): 
```
Recursively searches for modules to reload and then reloads them.
        Uses a cache to break cyclic dependencies of any sort
- `:returns`: `_`
    >No description...

### Examples


