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





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/TensorDerivativeConverter.py?message=Update%20Docs)