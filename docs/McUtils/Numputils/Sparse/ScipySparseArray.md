## <a id="McUtils.Numputils.Sparse.ScipySparseArray">ScipySparseArray</a>
Array class that generalize the regular `scipy.sparse.spmatrix`.
Basically acts like a high-dimensional wrapper that manages the _shape_ of a standard `scipy.sparse_matrix`, since that is rigidly 2D.
We always use a combo of an underlying CSR or CSC matrix & COO-like shape operations.

### Properties and Methods
```python
formats_map: dict
default_cache_size: int
caching_enabled: bool
```
<a id="McUtils.Numputils.Sparse.ScipySparseArray.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, a, shape=None, layout=None, dtype=None, initialize=True, cache_block_data=True, logger=None): 
```

- `a`: `Any`
    >No description...
- `shape`: `Any`
    >No description...
- `layout`: `Any`
    >No description...
- `dtype`: `Any`
    >No description...
- `initialize`: `Any`
    >No description...
- `cache_block_data`: `Any`
    >whether or not
- `logger`: `Logger`
    >the logger to use for debug purposes

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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.from_state" class="docs-object-method">&nbsp;</a>
```python
from_state(state, serializer=None): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.initialize_empty" class="docs-object-method">&nbsp;</a>
```python
initialize_empty(shape, dtype=None, layout=None, **kw): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.construct_sparse_from_val_inds" class="docs-object-method">&nbsp;</a>
```python
construct_sparse_from_val_inds(block_vals, block_inds, shape, fmt, cache_block_data=True, logger=None): 
```

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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.from_diagonal_data" class="docs-object-method">&nbsp;</a>
```python
from_diagonal_data(diags, shape=None, **kw): 
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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascsc" class="docs-object-method">&nbsp;</a>
```python
ascsc(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.data" class="docs-object-method">&nbsp;</a>
```python
@property
data(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.format_from_string" class="docs-object-method">&nbsp;</a>
```python
format_from_string(fmt): 
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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.get_caching_status" class="docs-object-method">&nbsp;</a>
```python
get_caching_status(): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.enable_caches" class="docs-object-method">&nbsp;</a>
```python
enable_caches(): 
```
A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this on
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.disable_caches" class="docs-object-method">&nbsp;</a>
```python
disable_caches(): 
```
A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this off
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.clear_cache" class="docs-object-method">&nbsp;</a>
```python
clear_cache(): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.clear_ravel_caches" class="docs-object-method">&nbsp;</a>
```python
clear_ravel_caches(): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.set_ravel_cache_size" class="docs-object-method">&nbsp;</a>
```python
set_ravel_cache_size(size): 
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
Had to make this op not in-place because otherwise got scary errors...
- `shp`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.squeeze" class="docs-object-method">&nbsp;</a>
```python
squeeze(self): 
```

<a id="McUtils.Numputils.Sparse.ScipySparseArray.resize" class="docs-object-method">&nbsp;</a>
```python
resize(self, newsize): 
```
Returns a resized version of the tensor
- `newsize`: `tuple[int]`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate" class="docs-object-method">&nbsp;</a>
```python
concatenate(self, *others, axis=0): 
```
Concatenates multiple arrays along the specified axis
        This is relatively inefficient in terms of not tracking indices
        throughout
- `other`: `Any`
    >No description...
- `axis`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.broadcast_to" class="docs-object-method">&nbsp;</a>
```python
broadcast_to(self, shape): 
```
Implements broadcast_to using COO-style operations
        to be a little bit more efficient
- `shape`: `Any`
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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.true_multiply" class="docs-object-method">&nbsp;</a>
```python
true_multiply(self, other): 
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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.loadz" class="docs-object-method">&nbsp;</a>
```python
loadz(file): 
```
Loads a SparseArray from an npz file
- `file`: `Any`
    >No description...
- `:returns`: `SparseArray`
    >No description...

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





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Numputils/Sparse/ScipySparseArray.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Numputils/Sparse/ScipySparseArray.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Numputils/Sparse/ScipySparseArray.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Numputils/Sparse/ScipySparseArray.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/Sparse.py?message=Update%20Docs)