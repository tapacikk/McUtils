## <a id="McUtils.Numputils.Sparse.ScipySparseArray">ScipySparseArray</a>
Array class that generalize the regular `scipy.sparse.spmatrix`.
Basically acts like a high-dimensional wrapper that manages the _shape_ of a standard `scipy.sparse_matrix`, since that is rigidly 2D.
We always a combo of an underlying CSR or CSC matrix & COO-like shape operations.

### Properties and Methods
```python
from_state: method
empty: method
from_diagonal_data: method
default_cache_size: int
caching_enabled: bool
get_caching_status: method
enable_caches: method
disable_caches: method
clear_cache: method
clear_ravel_caches: method
set_ravel_cache_size: method
loadz: method
```
<a id="McUtils.Numputils.Sparse.ScipySparseArray.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, a, shape=None, layout=<class 'scipy.sparse.csr.csr_matrix'>, dtype=None, initialize=True): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.to_state" class="docs-object-method">&nbsp;</a>
```python
to_state(self, serializer=None): 
```
Provides just the state that is needed to
        serialize the object
- `serializer`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.dtype" class="docs-object-method">&nbsp;</a>
```python
@property
dtype(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.diag" class="docs-object-method">&nbsp;</a>
```python
@property
diag(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.asarray" class="docs-object-method">&nbsp;</a>
```python
asarray(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.todense" class="docs-object-method">&nbsp;</a>
```python
todense(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascoo" class="docs-object-method">&nbsp;</a>
```python
ascoo(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascsr" class="docs-object-method">&nbsp;</a>
```python
ascsr(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.data" class="docs-object-method">&nbsp;</a>
```python
@property
data(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.fmt" class="docs-object-method">&nbsp;</a>
```python
@property
fmt(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.shape" class="docs-object-method">&nbsp;</a>
```python
@property
shape(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ndim" class="docs-object-method">&nbsp;</a>
```python
@property
ndim(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.non_zero_count" class="docs-object-method">&nbsp;</a>
```python
@property
non_zero_count(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.find" class="docs-object-method">&nbsp;</a>
```python
find(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_vals" class="docs-object-method">&nbsp;</a>
```python
@property
block_vals(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_inds" class="docs-object-method">&nbsp;</a>
```python
@property
block_inds(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_data" class="docs-object-method">&nbsp;</a>
```python
@property
block_data(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.profiled_transpose" class="docs-object-method">&nbsp;</a>
```python
profiled_transpose(self, transp): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.transpose" class="docs-object-method">&nbsp;</a>
```python
transpose(self, transp): 
```
Transposes the array and returns a new one.
        Not necessarily a cheap operation.
- `transp`: `Iterable[int]`
    >the transposition to do
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.reshape" class="docs-object-method">&nbsp;</a>
```python
reshape(self, shp): 
```
Had to make this op not in-plae because otherwise got scary errors...
- `shp`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.squeeze" class="docs-object-method">&nbsp;</a>
```python
squeeze(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate" class="docs-object-method">&nbsp;</a>
```python
concatenate(self, other, axis=0): 
```
Concatenates two arrays along the specified axis
- `other`: `Any`
    >No description...
- `axis`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.T" class="docs-object-method">&nbsp;</a>
```python
@property
T(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__matmul__" class="docs-object-method">&nbsp;</a>
```python
__matmul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.dot" class="docs-object-method">&nbsp;</a>
```python
dot(self, b, reverse=False): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__neg__" class="docs-object-method">&nbsp;</a>
```python
__neg__(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__pos__" class="docs-object-method">&nbsp;</a>
```python
__pos__(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__add__" class="docs-object-method">&nbsp;</a>
```python
__add__(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__radd__" class="docs-object-method">&nbsp;</a>
```python
__radd__(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.plus" class="docs-object-method">&nbsp;</a>
```python
plus(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.floopy_flop" class="docs-object-method">&nbsp;</a>
```python
floopy_flop(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__truediv__" class="docs-object-method">&nbsp;</a>
```python
__truediv__(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a>
```python
__rtruediv__(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__rmul__" class="docs-object-method">&nbsp;</a>
```python
__rmul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__mul__" class="docs-object-method">&nbsp;</a>
```python
__mul__(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.multiply" class="docs-object-method">&nbsp;</a>
```python
multiply(self, other): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.copy" class="docs-object-method">&nbsp;</a>
```python
copy(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.savez" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, item): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__setitem__" class="docs-object-method">&nbsp;</a>
```python
__setitem__(self, item, val): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__delitem__" class="docs-object-method">&nbsp;</a>
```python
__delitem__(self, item): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

### Examples


