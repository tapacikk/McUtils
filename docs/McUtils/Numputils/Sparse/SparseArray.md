## <a id="McUtils.Numputils.Sparse.SparseArray">SparseArray</a>
Array class that generalize the regular `scipy.sparse.spmatrix`.
Basically acts like a high-dimensional wrapper that manages the _shape_ of a standard `scipy.sparse_matrix`, since that is rigidly 2D.

### Properties and Methods
```python
from_state: method
empty: method
from_diag: method
clear_ravel_caches: method
loadz: method
```
<a id="McUtils.Numputils.Sparse.SparseArray.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, a, shape=None, layout=<class 'scipy.sparse.csc.csc_matrix'>, dtype=None, initialize=True): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.to_state" class="docs-object-method">&nbsp;</a>
```python
to_state(self, serializer=None): 
```
Provides just the state that is needed to
        serialize the object
- `serializer`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.dtype" class="docs-object-method">&nbsp;</a>
```python
@property
dtype(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.diag" class="docs-object-method">&nbsp;</a>
```python
@property
diag(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.toarray" class="docs-object-method">&nbsp;</a>
```python
toarray(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.todense" class="docs-object-method">&nbsp;</a>
```python
todense(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.data" class="docs-object-method">&nbsp;</a>
```python
@property
data(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.fmt" class="docs-object-method">&nbsp;</a>
```python
@property
fmt(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.shape" class="docs-object-method">&nbsp;</a>
```python
@property
shape(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.ndim" class="docs-object-method">&nbsp;</a>
```python
@property
ndim(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.non_zero_count" class="docs-object-method">&nbsp;</a>
```python
@property
non_zero_count(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.block_vals" class="docs-object-method">&nbsp;</a>
```python
@property
block_vals(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.block_inds" class="docs-object-method">&nbsp;</a>
```python
@property
block_inds(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.block_data" class="docs-object-method">&nbsp;</a>
```python
@property
block_data(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.transpose" class="docs-object-method">&nbsp;</a>
```python
transpose(self, transp): 
```
Transposes the array and returns a new one.
        Not necessarily a cheap operation.
- `transp`: `Iterable[int]`
    >the transposition to do
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.reshape" class="docs-object-method">&nbsp;</a>
```python
reshape(self, shp): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.squeeze" class="docs-object-method">&nbsp;</a>
```python
squeeze(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.T" class="docs-object-method">&nbsp;</a>
```python
@property
T(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__matmul__" class="docs-object-method">&nbsp;</a>
```python
__matmul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.dot" class="docs-object-method">&nbsp;</a>
```python
dot(self, b, reverse=False): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.tensordot" class="docs-object-method">&nbsp;</a>
```python
tensordot(self, b, axes=2, reverse=False): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__neg__" class="docs-object-method">&nbsp;</a>
```python
__neg__(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__pos__" class="docs-object-method">&nbsp;</a>
```python
__pos__(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__add__" class="docs-object-method">&nbsp;</a>
```python
__add__(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__radd__" class="docs-object-method">&nbsp;</a>
```python
__radd__(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.plus" class="docs-object-method">&nbsp;</a>
```python
plus(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.floopy_flop" class="docs-object-method">&nbsp;</a>
```python
floopy_flop(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__truediv__" class="docs-object-method">&nbsp;</a>
```python
__truediv__(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a>
```python
__rtruediv__(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__rmul__" class="docs-object-method">&nbsp;</a>
```python
__rmul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__mul__" class="docs-object-method">&nbsp;</a>
```python
__mul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.multiply" class="docs-object-method">&nbsp;</a>
```python
multiply(self, other): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.copy" class="docs-object-method">&nbsp;</a>
```python
copy(self): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.savez" class="docs-object-method">&nbsp;</a>
```python
savez(self, file, compressed=True): 
```
Saves a SparseArray to a file (must have the npz extension)
- `file`: `Any`
    >No description...
- `compressed`: `Any`
    >No description...
- `:returns`: `str`
    >the saved file

<a id="McUtils.Numputils.Sparse.SparseArray.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, item): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__setitem__" class="docs-object-method">&nbsp;</a>
```python
__setitem__(self, item, val): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__delitem__" class="docs-object-method">&nbsp;</a>
```python
__delitem__(self, item): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

### Examples


