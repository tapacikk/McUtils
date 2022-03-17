## <a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction">FiniteDifferenceFunction</a>
The FiniteDifferenceFunction encapsulates a bunch of functionality extracted from [Fornberger's
Calculation of Wieghts in Finite Difference Formulas](https://epubs.siam.org/doi/pdf/10.1137/S0036144596322507)

Only applies to direct product grids, but each subgrid can be regular or irregular.
Used in a large number of other places, but relatively rarely on its own.
A convenient application is the `FiniteDifferenceDerivative` class in the `Derivatives` module.

### Properties and Methods
<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *diffs, axes=0, contract=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L35)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L35?message=Update%20Docs)]
</div>

Constructs an object to take finite differences derivatives of grids of data
- `diffs`: `FiniteDifference1D`
    >A set of differences to take along successive axes in the data
- `axes`: `int | Iterable[int]`
    >The axes to take the specified differences along
- `contract`: `bool`
    >Whether to reduce the shape of the returned tensor if applicable after application

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.apply" class="docs-object-method">&nbsp;</a> 
```python
apply(self, vals, axes=None, mesh_spacing=None, contract=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L49)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L49?message=Update%20Docs)]
</div>

Iteratively applies the stored finite difference objects to the vals
- `vals`: `np.ndarray`
    >The tensor of values to take the difference on
- `axes`: `int | Iterable[int]`
    >The axis or axes to take the differences along (defaults to `self.axes`)
- `:returns`: `np.ndarray`
    >The tensor of derivatives

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, vals, axes=None, mesh_spacing=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L83)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L83?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.order" class="docs-object-method">&nbsp;</a> 
```python
@property
order(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L?message=Update%20Docs)]
</div>


- `:returns`: `tuple[int]`
    >the order of the derivative requested

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.weights" class="docs-object-method">&nbsp;</a> 
```python
@property
weights(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L?message=Update%20Docs)]
</div>


- `:returns`: `tuple[np.array[float]]`
    >the weights for the specified stencil

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.widths" class="docs-object-method">&nbsp;</a> 
```python
@property
widths(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L?message=Update%20Docs)]
</div>


- `:returns`: `tuple[(int, int)]`
    >the number of points in each dimension, left and right, for the specified stencil

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.regular_difference" class="docs-object-method">&nbsp;</a> 
```python
regular_difference(order, mesh_spacing=None, accuracy=2, stencil=None, end_point_accuracy=2, axes=0, contract=True, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L110)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L110?message=Update%20Docs)]
</div>

Constructs a `FiniteDifferenceFunction` appropriate for a _regular grid_ with the given stencil
- `order`: `tuple[int]`
    >the order of the derivative
- `mesh_spacing`: `None | float | tuple[float]`
    >the spacing between grid points in the regular grid `h`
- `accuracy`: `None | int | tuple[int]`
    >the accuracy of the derivative that we'll try to achieve as a power on `h`
- `stencil`: `None | int | tuple[int]`
    >the stencil to use for the derivative (overrides `accuracy`)
- `end_point_accuracy`: `None | int | tuple[int]`
    >the amount of extra accuracy to use at the edges of the grid
- `axes`: `None | int | tuple[int]`
    >the axes of the passed array for the derivative to be applied along
- `contract`: `bool`
    >whether to eliminate any axes of size `1` from the results
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.from_grid" class="docs-object-method">&nbsp;</a> 
```python
from_grid(grid, order, accuracy=2, stencil=None, end_point_accuracy=2, axes=0, contract=True, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L168)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FiniteDifferenceFunction.py#L168?message=Update%20Docs)]
</div>

Constructs a `FiniteDifferenceFunction` from a grid and order.
         Deconstructs the grid into its subgrids and builds a different differencer for each dimension
- `grid`: `np.ndarray`
    >The grid to use as input data when defining the derivative
- `order`: `int or list of ints`
    >order of the derivative to compute
- `stencil`: `int or list of ints`
    >number of points to use in the stencil
- `:returns`: `FiniteDifferenceFunction`
    >deriv func

## <a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction">FiniteDifferenceFunction</a>
The FiniteDifferenceFunction encapsulates a bunch of functionality extracted from Fornberger's
Calculation of Wieghts in Finite Difference Formulas (https://epubs.siam.org/doi/pdf/10.1137/S0036144596322507)

Only applies to direct product grids, but each subgrid can be regular or irregular

### Properties and Methods
```python
regular_difference: method
from_grid: method
```
<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, *diffs, axes=0, contract=False): 
```
Constructs an object to take finite differences derivatives of grids of data
- `diffs`: `FiniteDifference1D`
    >A set of differences to take along successive axes in the data
- `axes`: `int | Iterable[int]`
    >The axes to take the specified differences along
- `contract`: `bool`
    >Whether to reduce the shape of the returned tensor if applicable after application

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, vals, axes=None, mesh_spacing=None, contract=None): 
```
Iteratively applies the stored finite difference objects to the vals
- `vals`: `np.ndarray`
    >The tensor of values to take the difference on
- `axes`: `int | Iterable[int]`
    >The axis or axes to take the differences along (defaults to `self.axes`)
- `:returns`: `np.ndarray`
    >The tensor of derivatives

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, vals, axes=None, mesh_spacing=None): 
```

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.order" class="docs-object-method">&nbsp;</a>
```python
@property
order(self): 
```

- `:returns`: `tuple[int]`
    >the order of the derivative requested

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.weights" class="docs-object-method">&nbsp;</a>
```python
@property
weights(self): 
```

- `:returns`: `tuple[np.array[float]]`
    >the weights for the specified stencil

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.FiniteDifferenceFunction.widths" class="docs-object-method">&nbsp;</a>
```python
@property
widths(self): 
```

- `:returns`: `tuple[(int, int)]`
    >the number of points in each dimension, left and right, for the specified stencil

### Examples


___

[Edit Examples](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) or 
[Create New Examples](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) <br/>
[Edit Template](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/templates/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) or 
[Create New Template](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/templates/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py?message=Update%20Docs)



___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Zachary/Taylor/FiniteDifferenceFunction/FiniteDifferenceFunction.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Taylor/FiniteDifferenceFunction.py?message=Update%20Docs)