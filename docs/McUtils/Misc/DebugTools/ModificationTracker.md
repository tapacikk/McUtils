## <a id="McUtils.Misc.DebugTools.ModificationTracker">ModificationTracker</a>
A simple class to wrap an object to track when it is accessed or
modified

### Properties and Methods
<a id="McUtils.Misc.DebugTools.ModificationTracker.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, obj, handlers=<ModificationTypeHandler.Log: 'log'>, logger=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L32)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L32?message=Update%20Docs)]
</div>

<a id="McUtils.Misc.DebugTools.ModificationTracker.handler_dispatch" class="docs-object-method">&nbsp;</a> 
```python
@property
handler_dispatch(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Misc.DebugTools.ModificationTracker.log_modification" class="docs-object-method">&nbsp;</a> 
```python
log_modification(self, obj, handler_type, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L51)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L51?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L81)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L81?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L125)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L125?message=Update%20Docs)]
</div>

Handler to intercept `getattr` requests
- `item`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.DebugTools.ModificationTracker.__setattr__" class="docs-object-method">&nbsp;</a> 
```python
__setattr__(self, item, val): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L137)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L137?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L152)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L152?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L167)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L167?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L182)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L182?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L197)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L197?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Misc/DebugTools.py#L212)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Misc/DebugTools.py#L212?message=Update%20Docs)]
</div>

Handler to intercept `matmul` requests
- `item`: `Any`
    >No description...
- `val`: `Any`
    >No description...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Misc/DebugTools/ModificationTracker.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Misc/DebugTools/ModificationTracker.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Misc/DebugTools/ModificationTracker.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Misc/DebugTools/ModificationTracker.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Misc/DebugTools.py?message=Update%20Docs)