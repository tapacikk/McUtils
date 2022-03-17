## <a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface">BaseSurface</a>
Surface base class which can be subclassed for relevant cases

### Properties and Methods
<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, data, dimension): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L19)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L19?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, points, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L24)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L24?message=Update%20Docs)]
</div>

Evaluates the function at the points based off of "data"
- `points`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, gridpoints, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L35)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L35?message=Update%20Docs)]
</div>


- `gridpoints`: `np.ndarray`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.minimize" class="docs-object-method">&nbsp;</a> 
```python
minimize(self, initial_guess, function_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L64)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L64?message=Update%20Docs)]
</div>

Just calls into `scipy.optimize.minimize` as the default implementation
- `initial_guess`: `np.ndarray`
    >starting position for the minimzation
- `function_options`: `dict | None`
    >No description...
- `opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Surfaces/BaseSurface.py?message=Update%20Docs)