## <a id="McUtils.Zachary.TensorDerivativeConverter.TensorDerivativeConverter">TensorDerivativeConverter</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter.py#L707)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter.py#L707?message=Update%20Docs)]
</div>

A class that makes it possible to convert expressions
involving derivatives in one coordinate system in another

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
TensorExpansionError: type
```
<a id="McUtils.Zachary.TensorDerivativeConverter.TensorDerivativeConverter.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, jacobians, derivatives=None, mixed_terms=None, jacobians_name='Q', values_name='V'): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter.py#L716)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter.py#L716?message=Update%20Docs)]
</div>


- `jacobians`: `Iterable[np.ndarray]`
    >The Jacobian and higher-order derivatives between the coordinate systems
- `derivatives`: `Iterable[np.ndarray]`
    >Derivatives of some quantity in the original coordinate system
- `mixed_terms`: `Iterable[Iterable[None | np.ndarray]]`
    >Mixed derivatives of some quantity involving the new and old coordinates

<a id="McUtils.Zachary.TensorDerivativeConverter.TensorDerivativeConverter.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, order=None, print_transformations=False, check_arrays=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter.py#L736)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter.py#L736?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/TensorDerivativeConverter/TensorDerivativeConverter.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter.py#L707?message=Update%20Docs)