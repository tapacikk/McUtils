## <a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.RegularGridFiniteDifference">RegularGridFiniteDifference</a>
Defines a 1D finite difference over a regular grid

### Properties and Methods
<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.RegularGridFiniteDifference.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, order, stencil=None, accuracy=4, end_point_accuracy=2, **kw): 
```

- `order`: `int`
    >the order of the derivative to take
- `stencil`: `int | None`
    >the number of stencil points to add
- `accuracy`: `int | None`
    >the approximate accuracy to target with the method
- `end_point_accuracy`: `int | None`
    >the extra number of stencil points to add to the end points
- `kw`: `Any`
    >options passed through to the `FiniteDifferenceMatrix`

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.RegularGridFiniteDifference.finite_difference_data" class="docs-object-method">&nbsp;</a>
```python
finite_difference_data(order, stencil, end_point_precision): 
```
Builds a FiniteDifferenceData object from an order, stencil, and end_point_precision
- `order`: `Any`
    >No description...
- `stencil`: `Any`
    >No description...
- `end_point_precision`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.RegularGridFiniteDifference.get_weights" class="docs-object-method">&nbsp;</a>
```python
get_weights(m, s, n): 
```
Extracts the weights for an evenly spaced grid
- `m`: `Any`
    >No description...
- `s`: `Any`
    >No description...
- `n`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/Taylor/FiniteDifferenceFunction/RegularGridFiniteDifference.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/Taylor/FiniteDifferenceFunction/RegularGridFiniteDifference.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/Taylor/FiniteDifferenceFunction/RegularGridFiniteDifference.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/Taylor/FiniteDifferenceFunction/RegularGridFiniteDifference.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Taylor/FiniteDifferenceFunction.py?message=Update%20Docs)