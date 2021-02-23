## <a id="McUtils.Numputils.Sparse.TensorFlowSparseArray">TensorFlowSparseArray</a>
Provides a SparseArray implementation that uses TensorFlow as the backend

### Properties and Methods
```python
from_state: method
empty: method
```
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

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.multiply" class="docs-object-method">&nbsp;</a>
```python
multiply(self, other): 
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


