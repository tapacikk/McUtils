## <a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface">LinearFitSurface</a>
A surface built off of a LinearExpansionSurface, but done by fitting.
The basis selection

### Properties and Methods
<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, points, basis=None, order=4, dimension=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L160)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L160?message=Update%20Docs)]
</div>


- `points`: `np.ndarray`
    >a set of points to fit to
- `basis`: `Iterable[function] | None`
    >a basis of functions to use (defaults to power series)

<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, points, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L182)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L182?message=Update%20Docs)]
</div>


- `points`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface.minimize" class="docs-object-method">&nbsp;</a> 
```python
minimize(self, initial_guess=None, function_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L196)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L196?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Surfaces/BaseSurface.py?message=Update%20Docs)