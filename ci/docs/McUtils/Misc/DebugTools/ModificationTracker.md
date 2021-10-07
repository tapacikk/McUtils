## <a id="McUtils.Misc.DebugTools.ModificationTracker">ModificationTracker</a>
A simple class to wrap an object to track when it is accessed or
modified

### Properties and Methods
<a id="McUtils.Misc.DebugTools.ModificationTracker.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, obj, handlers=<ModificationTypeHandler.Log: 'log'>, logger=None): 
```

<a id="McUtils.Misc.DebugTools.ModificationTracker.handler_dispatch" class="docs-object-method">&nbsp;</a>
```python
@property
handler_dispatch(self): 
```

<a id="McUtils.Misc.DebugTools.ModificationTracker.log_modification" class="docs-object-method">&nbsp;</a>
```python
log_modification(self, obj, handler_type, *args, **kwargs): 
```
Logs on modification
- `obj`: `Any`
    >No description...
- `handler_type`: `Any`
    >No description...
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.raise_modification" class="docs-object-method">&nbsp;</a>
```python
raise_modification(self, obj, handler_type, *args, **kwargs): 
```
Raises an error on modification
- `obj`: `Any`
    >No description...
- `handler_type`: `Any`
    >No description...
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__getattr__" class="docs-object-method">&nbsp;</a>
```python
__getattr__(self, item): 
```
Handler to intercept `getattr` requests
- `item`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__setattr__" class="docs-object-method">&nbsp;</a>
```python
__setattr__(self, item, val): 
```
Handler to intercept `setattr` requests
- `item`: `Any`
    >No description...
- `val`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__iadd__" class="docs-object-method">&nbsp;</a>
```python
__iadd__(self, other): 
```
Handler to intercept `add` requests
- `item`: `Any`
    >No description...
- `val`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__isub__" class="docs-object-method">&nbsp;</a>
```python
__isub__(self, other): 
```
Handler to intercept `sub` requests
- `item`: `Any`
    >No description...
- `val`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__imul__" class="docs-object-method">&nbsp;</a>
```python
__imul__(self, other): 
```
Handler to intercept `div` requests
- `item`: `Any`
    >No description...
- `val`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__idiv__" class="docs-object-method">&nbsp;</a>
```python
__idiv__(self, other): 
```
Handler to intercept `div` requests
- `item`: `Any`
    >No description...
- `val`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__imatmul__" class="docs-object-method">&nbsp;</a>
```python
__imatmul__(self, other): 
```
Handler to intercept `matmul` requests
- `item`: `Any`
    >No description...
- `val`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Misc/DebugTools/ModificationTracker.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Misc/DebugTools/ModificationTracker.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Misc/DebugTools/ModificationTracker.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Misc/DebugTools/ModificationTracker.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Misc/DebugTools.py?message=Update%20Docs)