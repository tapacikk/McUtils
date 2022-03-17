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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/InteractiveTools.py#L17)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/InteractiveTools.py#L17?message=Update%20Docs)]
</div>


- `modspec`: `str | types.ModuleType`
    >No description...

<a id="McUtils.Misc.InteractiveTools.ModuleReloader.get_parents" class="docs-object-method">&nbsp;</a> 
```python
get_parents(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/InteractiveTools.py#L26)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/InteractiveTools.py#L26?message=Update%20Docs)]
</div>

Returns module parents
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.InteractiveTools.ModuleReloader.get_members" class="docs-object-method">&nbsp;</a> 
```python
get_members(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/InteractiveTools.py#L35)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/InteractiveTools.py#L35?message=Update%20Docs)]
</div>

Returns module members
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.InteractiveTools.ModuleReloader.reload_member" class="docs-object-method">&nbsp;</a> 
```python
reload_member(self, member, stack=None, reloaded=None, blacklist=None, reload_parents=True, verbose=False, print_indent=''): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/InteractiveTools.py#L52)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/InteractiveTools.py#L52?message=Update%20Docs)]
</div>

<a id="McUtils.Misc.InteractiveTools.ModuleReloader.reload" class="docs-object-method">&nbsp;</a> 
```python
reload(self, stack=None, reloaded=None, blacklist=None, reload_parents=True, verbose=False, print_indent=''): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/InteractiveTools.py#L105)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/InteractiveTools.py#L105?message=Update%20Docs)]
</div>

Recursively searches for modules to reload and then reloads them.
        Uses a cache to break cyclic dependencies of any sort.
        This turns out to also be a challenging problem, since we need to basically
        load depth-first, while never jumping too far back...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Misc/InteractiveTools/ModuleReloader.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Misc/InteractiveTools/ModuleReloader.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Misc/InteractiveTools/ModuleReloader.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Misc/InteractiveTools/ModuleReloader.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Misc/InteractiveTools.py?message=Update%20Docs)