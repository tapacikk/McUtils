## <a id="McUtils.Zachary.FittableModels.LinearFitBasis">LinearFitBasis</a>
Provides a container to build bases of functions for fitting.
Asks for a generator for each dimension, which is just a function that takes an integer and returns a basis function at that order.
Product functions are taken up to some max order

### Properties and Methods
<a id="McUtils.Zachary.FittableModels.LinearFitBasis.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, *generators, order=3): 
```

- `generators`: `Iterable[function]`
    >the generating functions for the bases in each dimenion
- `order`: `int`
    >the maximum order for the basis functions (currently turning off coupling isn't possible, but that could come)

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.functions" class="docs-object-method">&nbsp;</a>
```python
@property
functions(self): 
```

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.names" class="docs-object-method">&nbsp;</a>
```python
@property
names(self): 
```

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.order" class="docs-object-method">&nbsp;</a>
```python
@property
order(self): 
```

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.construct_basis" class="docs-object-method">&nbsp;</a>
```python
construct_basis(self): 
```

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.<lambda>" class="docs-object-method">&nbsp;</a>
```python
fourier_series(x, k): 
```

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.<lambda>" class="docs-object-method">&nbsp;</a>
```python
power_series(x, n): 
```

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/FittableModels/LinearFitBasis.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/FittableModels/LinearFitBasis.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/FittableModels/LinearFitBasis.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/FittableModels/LinearFitBasis.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/FittableModels.py?message=Update%20Docs)