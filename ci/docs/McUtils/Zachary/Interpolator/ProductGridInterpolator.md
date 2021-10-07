## <a id="McUtils.Zachary.Interpolator.ProductGridInterpolator">ProductGridInterpolator</a>
A set of interpolators that support interpolation
on a regular (tensor product) grid

### Properties and Methods
<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, grids, vals, caller=None, order=None, extrapolate=True): 
```

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

- `order`: `Any`
    >No description...
- `:returns`: `ProductGridInterpolator`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/Interpolator/ProductGridInterpolator.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Interpolator.py?message=Update%20Docs)