## <a id="McUtils.Zachary.TensorDerivativeConverter.TensorDerivativeConverter">TensorDerivativeConverter</a>
A class that makes it possible to convert expressions
involving derivatives in one coordinate system in another

### Properties and Methods
```python
TensorExpansionError: type
```
<a id="McUtils.Zachary.TensorDerivativeConverter.TensorDerivativeConverter.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, jacobians, derivatives=None, mixed_terms=None, jacobians_name='Q', values_name='V'): 
```

- `jacobians`: `Iterable[np.ndarray]`
    >The Jacobian and higher-order derivatives between the coordinate systems
- `derivatives`: `Iterable[np.ndarray]`
    >Derivatives of some quantity in the original coordinate system
- `mixed_terms`: `Iterable[Iterable[None | np.ndarray]]`
    >Mixed derivatives of some quantity involving the new and old coordinates

<a id="McUtils.Zachary.TensorDerivativeConverter.TensorDerivativeConverter.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, order=None, check_arrays=False): 
```

### Examples


