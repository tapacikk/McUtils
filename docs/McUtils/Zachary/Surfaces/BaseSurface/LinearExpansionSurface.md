## <a id="McUtils.Zachary.Surfaces.BaseSurface.LinearExpansionSurface">LinearExpansionSurface</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L123)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L123?message=Update%20Docs)]
</div>

A surface with an evaluator built off of an expansion in some user specified basis

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearExpansionSurface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, coefficients, basis=None, dimension=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L127)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L127?message=Update%20Docs)]
</div>


- `coefficients`: `np.ndarray`
    >the expansion coefficients in the basis
- `basis`: `Iterable[function] | None`
    >a basis of functions to use (defaults to power series)

<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearExpansionSurface.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, points, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L140)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L140?message=Update%20Docs)]
</div>

First we just apply the basis to the gridpoints, then we dot this into the coeffs
- `points`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Surfaces/BaseSurface/LinearExpansionSurface.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Surfaces/BaseSurface/LinearExpansionSurface.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Surfaces/BaseSurface/LinearExpansionSurface.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/BaseSurface/LinearExpansionSurface.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L123?message=Update%20Docs)