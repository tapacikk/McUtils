## <a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion">FunctionExpansion</a>
A class for handling expansions of an internal coordinate potential up to 4th order
Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian

### Properties and Methods
```python
CoordinateTransforms: type
FunctionDerivatives: type
```
<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, derivatives, transforms=None, center=None, ref=0, weight_coefficients=True): 
```

- `derivatives`: `Iterable[np.ndarray | Tensor]`
    >Derivatives of the function being expanded
- `transforms`: `Iterable[np.ndarray | Tensor] | None`
    >Jacobian and higher order derivatives in the coordinates
- `center`: `np.ndarray | None`
    >the reference point for expanding aobut
- `ref`: `float | np.ndarray`
    >the reference point value for shifting the expansion
- `weight_coefficients`: `bool`
    >whether the derivative terms need to be weighted or not

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand_function" class="docs-object-method">&nbsp;</a>
```python
expand_function(f, point, order=4, basis=None, function_shape=None, transforms=None, weight_coefficients=True, **fd_options): 
```
Expands a function about a point up to the given order
- `f`: `function`
    >No description...
- `point`: `np.ndarray | CoordinateSet`
    >No description...
- `order`: `int`
    >No description...
- `basis`: `None | CoordinateSystem`
    >No description...
- `fd_options`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expansion_tensors" class="docs-object-method">&nbsp;</a>
```python
@property
expansion_tensors(self): 
```
Provides the tensors that will contracted
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_expansions" class="docs-object-method">&nbsp;</a>
```python
get_expansions(self, coords, squeeze=True): 
```

- `coords`: `np.ndarray | CoordinateSet`
    >Coordinates to evaluate the expansion at
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand" class="docs-object-method">&nbsp;</a>
```python
expand(self, coords, squeeze=True): 
```
Returns a numerical value for the expanded coordinates
- `coords`: `np.ndarray`
    >No description...
- `:returns`: `float | np.ndarray`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, coords, **kw): 
```

## <a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion">FunctionExpansion</a>
A class for handling expansions of an internal coordinate potential up to 4th order
Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian

### Properties and Methods
```python
expand_function: method
CoordinateTransforms: type
FunctionDerivatives: type
```
<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, derivatives, transforms=None, center=None, ref=0, weight_coefficients=True): 
```

- `derivatives`: `Iterable[np.ndarray | Tensor]`
    >Derivatives of the function being expanded
- `transforms`: `Iterable[np.ndarray | Tensor] | None`
    >Jacobian and higher order derivatives in the coordinates
- `center`: `np.ndarray | None`
    >the reference point for expanding aobut
- `ref`: `float | np.ndarray`
    >the reference point value for shifting the expansion
- `weight_coefficients`: `bool`
    >whether the derivative terms need to be weighted or not

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.tensors" class="docs-object-method">&nbsp;</a>
```python
@property
tensors(self): 
```

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_expansions" class="docs-object-method">&nbsp;</a>
```python
get_expansions(self, coords, squeeze=True): 
```

- `coords`: `np.ndarray | CoordinateSet`
    >Coordinates to evaluate the expansion at
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand" class="docs-object-method">&nbsp;</a>
```python
expand(self, coords, squeeze=True): 
```
Returns a numerical value for the expanded coordinates
- `coords`: `np.ndarray`
    >No description...
- `:returns`: `float | np.ndarray`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, coords, **kw): 
```

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_tensor" class="docs-object-method">&nbsp;</a>
```python
get_tensor(self, i): 
```
Defines the overall tensors of derivatives
- `i`: `order of derivative tensor to provide`
    >No description...
- `:returns`: `Tensor`
    >No description...

### Examples


___

[Edit Examples](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Examples](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Template](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Template](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions.py?message=Update%20Docs)



___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Taylor/FunctionExpansions.py?message=Update%20Docs)