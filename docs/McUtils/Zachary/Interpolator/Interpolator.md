## <a id="McUtils.Zachary.Interpolator.Interpolator">Interpolator</a>
A general purpose that takes your data and just interpolates it without whining or making you do a pile of extra work

### Properties and Methods
```python
DefaultExtrapolator: ExtrapolatorType
```
<a id="McUtils.Zachary.Interpolator.Interpolator.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, grid, vals, interpolation_function=None, interpolation_order=None, extrapolator=None, extrapolation_order=None, **interpolation_opts): 
```

- `grid`: `np.ndarray`
    >an unstructured grid of points **or** a structured grid of points **or** a 1D array
- `vals`: `np.ndarray`
    >the values at the grid points
- `interpolation_function`: `None | BasicInterpolator`
    >the basic function to be used to handle the raw interpolation
- `interpolation_order`: `int | str | None`
    >the order of extrapolation to use (when applicable)
- `extrapolator`: `Extrapolator | None | str | function`
    >the extrapolator to use for data points not on the grid
- `extrapolation_order`: `int | str | None`
    >the order of extrapolation to use by default
- `interpolation_opts`: `Any`
    >the options to be fed into the interpolating_function

<a id="McUtils.Zachary.Interpolator.Interpolator.get_interpolator" class="docs-object-method">&nbsp;</a>
```python
get_interpolator(grid, vals, interpolation_order=None, allow_extrapolation=True, **opts): 
```
Returns a function that can be called on grid points to interpolate them
- `grid`: `Mesh`
    >No description...
- `vals`: `np.ndarray`
    >No description...
- `interpolation_order`: `int | str | None`
    >No description...
- `opts`: `Any`
    >No description...
- `:returns`: `function`
    >interpolator

<a id="McUtils.Zachary.Interpolator.Interpolator.get_extrapolator" class="docs-object-method">&nbsp;</a>
```python
get_extrapolator(grid, vals, extrapolation_order=1, **opts): 
```
Returns an Extrapolator that can be called on grid points to extrapolate them
- `grid`: `Mesh`
    >No description...
- `extrapolation_order`: `int`
    >No description...
- `:returns`: `Extrapolator`
    >extrapolator

<a id="McUtils.Zachary.Interpolator.Interpolator.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, grid_points, **opts): 
```
Interpolates then extrapolates the function at the grid_points
- `grid_points`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Interpolator.Interpolator.derivative" class="docs-object-method">&nbsp;</a>
```python
derivative(self, order): 
```
Returns a new function representing the requested derivative
        of the current interpolator
- `order`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Interpolator.Interpolator.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, *args, **kwargs): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/Interpolator/Interpolator.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/Interpolator/Interpolator.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/Interpolator/Interpolator.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/Interpolator/Interpolator.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Interpolator.py?message=Update%20Docs)