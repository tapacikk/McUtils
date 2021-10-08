## <a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpansionTerms">TensorExpansionTerms</a>
A friend of DumbTensor which exists
to not only make the tensor algebra suck less but also
to make it automated by making use of some simple rules
for expressing derivatives specifically in the context of
doing the coordinate transformations we need to do.
Everything here is 1 indexed since that's how I did the OG math

### Properties and Methods
```python
TensorExpansionTerm: ABCMeta
SumTerm: ABCMeta
ScalingTerm: ABCMeta
PowerTerm: ABCMeta
FlippedTerm: ABCMeta
AxisShiftTerm: ABCMeta
ContractionTerm: ABCMeta
QXTerm: ABCMeta
XVTerm: ABCMeta
QXVTerm: ABCMeta
BasicContractionTerm: ABCMeta
InverseTerm: ABCMeta
TraceTerm: ABCMeta
DeterminantTerm: ABCMeta
```
<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpansionTerms.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, qx_terms, xv_terms, qxv_terms=None, base_qx=None, base_xv=None, q_name='Q', v_name='V'): 
```

- `qx_terms`: `Iterable[np.ndarray]`
    >No description...
- `xv_terms`: `Iterable[np.ndarray]`
    >No description...

<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpansionTerms.QX" class="docs-object-method">&nbsp;</a>
```python
QX(self, n): 
```

<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpansionTerms.XV" class="docs-object-method">&nbsp;</a>
```python
XV(self, m): 
```

<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpansionTerms.QXV" class="docs-object-method">&nbsp;</a>
```python
QXV(self, n, m): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorExpansionTerms.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorExpansionTerms.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorExpansionTerms.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorExpansionTerms.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/TensorDerivativeConverter.py?message=Update%20Docs)