## <a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis">LinearFitBasis</a>
Provides a container to build bases of functions for fitting.
Asks for a generator for each dimension, which is just a function that takes an integer and returns a basis function at that order.
Product functions are taken up to some max order

### Properties and Methods
<a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, *generators, order=3): 
```

- `generators`: `Iterable[function]`
    >the generating functions for the bases in each dimenion
- `order`: `int`
    >the maximum order for the basis functions (currently turning off coupling isn't possible, but that could come)

<a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis.functions" class="docs-object-method">&nbsp;</a>
```python
@property
functions(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis.names" class="docs-object-method">&nbsp;</a>
```python
@property
names(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis.order" class="docs-object-method">&nbsp;</a>
```python
@property
order(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis.construct_basis" class="docs-object-method">&nbsp;</a>
```python
construct_basis(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis.<lambda>" class="docs-object-method">&nbsp;</a>
```python
fourier_series(x, k): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.LinearFitBasis.<lambda>" class="docs-object-method">&nbsp;</a>
```python
power_series(x, n): 
```

### Examples


