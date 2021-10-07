## <a id="McUtils.Numputils.Sparse.TensorFlowSparseArray">TensorFlowSparseArray</a>
Provides a SparseArray implementation that uses TensorFlow as the backend

### Properties and Methods
<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, data, dtype=None): 
```

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.shape" class="docs-object-method">&nbsp;</a>
```python
@property
shape(self): 
```
Provides the shape of the sparse array
- `:returns`: `tuple[int]`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.to_state" class="docs-object-method">&nbsp;</a>
```python
to_state(self, serializer=None): 
```
Provides just the state that is needed to
        serialize the object
- `serializer`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.from_state" class="docs-object-method">&nbsp;</a>
```python
from_state(state, serializer=None): 
```
Loads from the stored state
- `serializer`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.empty" class="docs-object-method">&nbsp;</a>
```python
empty(shape, dtype=None, **kw): 
```
Returns an empty SparseArray with the appropriate shape and dtype
- `shape`: `Any`
    >No description...
- `dtype`: `Any`
    >No description...
- `kw`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.block_data" class="docs-object-method">&nbsp;</a>
```python
@property
block_data(self): 
```
Returns the row and column indices and vector of
        values that the sparse array is storing
- `:returns`: `Tuple[np.ndarray, Iterable[np.ndarray]]`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.transpose" class="docs-object-method">&nbsp;</a>
```python
transpose(self, axes): 
```
Returns a transposed version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.ascoo" class="docs-object-method">&nbsp;</a>
```python
ascoo(self): 
```
Converts the tensor into a scipy COO matrix...
- `:returns`: `sp.coo_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.ascsr" class="docs-object-method">&nbsp;</a>
```python
ascsr(self): 
```
Converts the tensor into a scipy COO matrix...
- `:returns`: `sp.coo_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.reshape" class="docs-object-method">&nbsp;</a>
```python
reshape(self, newshape): 
```
Returns a reshaped version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__truediv__" class="docs-object-method">&nbsp;</a>
```python
__truediv__(self, other): 
```

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a>
```python
__rtruediv__(self, other): 
```

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__rmul__" class="docs-object-method">&nbsp;</a>
```python
__rmul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__mul__" class="docs-object-method">&nbsp;</a>
```python
__mul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.true_multiply" class="docs-object-method">&nbsp;</a>
```python
true_multiply(self, other): 
```
Multiplies self and other
- `other`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.dot" class="docs-object-method">&nbsp;</a>
```python
dot(self, other): 
```
Takes a regular dot product of self and other
- `other`: `Any`
    >No description...
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/Sparse.py?message=Update%20Docs)