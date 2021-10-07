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
reload(self, stack=None, reloaded=None, blacklist=None, reload_parents=True, verbose=False, print_indent=''): 
```
Recursively searches for modules to reload and then reloads them.
        Uses a cache to break cyclic dependencies of any sort.
        This turns out to also be a challenging problem, since we need to basically
        load depth-first, while never jumping too far back...
- `:returns`: `_`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Misc/InteractiveTools/ModuleReloader.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Misc/InteractiveTools/ModuleReloader.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Misc/InteractiveTools/ModuleReloader.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Misc/InteractiveTools/ModuleReloader.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Misc/InteractiveTools.py?message=Update%20Docs)