## <a id="McUtils.Zachary.Interpolator.ProductGridInterpolator">ProductGridInterpolator</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Interpolator.py#L52)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Interpolator.py#L52?message=Update%20Docs)]
</div>

A set of interpolators that support interpolation
on a regular (tensor product) grid

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, grids, vals, caller=None, order=None, extrapolate=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Interpolator.py#L58)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Interpolator.py#L58?message=Update%20Docs)]
</div>


- `grids`: `Any`
    >No description...
- `points`: `Any`
    >No description...
- `caller`: `Any`
    >No description...
- `order`: `int | Iterable[int]`
    >No description...

<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.construct_ndspline" class="docs-object-method">&nbsp;</a> 
```python
construct_ndspline(grids, vals, order, extrapolate=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Interpolator.py#L92)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Interpolator.py#L92?message=Update%20Docs)]
</div>

Builds a tensor product ndspline by constructing a product of 1D splines
- `grids`: `Iterable[np.ndarray]`
    >grids for each dimension independently
- `vals`: `np.ndarray`
    >No description...
- `order`: `int | Iterable[int]`
    >No description...
- `:returns`: `interpolate.NdPPoly`
    >No description...

<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Interpolator.py#L134)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Interpolator.py#L134?message=Update%20Docs)]
</div>


- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `np.ndarray`
    >No description...

<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.derivative" class="docs-object-method">&nbsp;</a> 
```python
derivative(self, order): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Interpolator.py#L145)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Interpolator.py#L145?message=Update%20Docs)]
</div>


- `order`: `Any`
    >No description...
- `:returns`: `ProductGridInterpolator`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Interpolator.py#L52?message=Update%20Docs)