## <a id="McUtils.Numputils.Sparse.SparseArray">SparseArray</a>
Represents a generic sparse array format
which can be subclassed to provide a concrete implementation

### Properties and Methods
```python
backends: NoneType
```
<a id="McUtils.Numputils.Sparse.SparseArray.get_backends" class="docs-object-method">&nbsp;</a>
```python
get_backends(): 
```
Provides the set of backends to try by default
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.from_data" class="docs-object-method">&nbsp;</a>
```python
from_data(data, shape=None, dtype=None, target_backend=None, constructor=None, **kwargs): 
```
A wrapper so that we can dispatch to the best
        sparse backend we've got defined.
        Can be monkey patched.
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `SparseArray`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.from_diag" class="docs-object-method">&nbsp;</a>
```python
from_diag(data, shape=None, dtype=None, **kwargs): 
```
A wrapper so that we can dispatch to the best
        sparse backend we've got defined.
        Can be monkey patched.
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.from_diagonal_data" class="docs-object-method">&nbsp;</a>
```python
from_diagonal_data(diags, **kw): 
```
Constructs a sparse tensor from diagonal elements
- `diags`: `Any`
    >No description...
- `kw`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.shape" class="docs-object-method">&nbsp;</a>
```python
@property
shape(self): 
```
Provides the shape of the sparse array
- `:returns`: `tuple[int]`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.ndim" class="docs-object-method">&nbsp;</a>
```python
@property
ndim(self): 
```
Provides the number of dimensions in the array
- `:returns`: `_`
    >No description...

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

<a id="McUtils.Numputils.Sparse.SparseArray.from_state" class="docs-object-method">&nbsp;</a>
```python
from_state(state, serializer=None): 
```
Loads from the stored state
- `serializer`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.empty" class="docs-object-method">&nbsp;</a>
```python
empty(shape, dtype=None, **kw): 
```

<a id="McUtils.Numputils.Sparse.SparseArray.initialize_empty" class="docs-object-method">&nbsp;</a>
```python
initialize_empty(shp, shape=None, **kw): 
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

<a id="McUtils.Numputils.Sparse.SparseArray.block_data" class="docs-object-method">&nbsp;</a>
```python
@property
block_data(self): 
```
Returns the vector of values and corresponding indices
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.block_inds" class="docs-object-method">&nbsp;</a>
```python
@property
block_inds(self): 
```
Returns indices for the stored values
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.transpose" class="docs-object-method">&nbsp;</a>
```python
transpose(self, axes): 
```
Returns a transposed version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.ascoo" class="docs-object-method">&nbsp;</a>
```python
ascoo(self): 
```
Converts the tensor into a scipy COO matrix...
- `:returns`: `sp.coo_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.ascsr" class="docs-object-method">&nbsp;</a>
```python
ascsr(self): 
```
Converts the tensor into a scipy CSR matrix...
- `:returns`: `sp.csr_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.asarray" class="docs-object-method">&nbsp;</a>
```python
asarray(self): 
```
Converts the tensor into a dense np.ndarray
- `:returns`: `np.ndarray`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.reshape" class="docs-object-method">&nbsp;</a>
```python
reshape(self, newshape): 
```
Returns a reshaped version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.resize" class="docs-object-method">&nbsp;</a>
```python
resize(self, newsize): 
```
Returns a resized version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.expand_dims" class="docs-object-method">&nbsp;</a>
```python
expand_dims(self, axis): 
```
adapted from np.expand_dims
- `axis`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.moveaxis" class="docs-object-method">&nbsp;</a>
```python
moveaxis(self, start, end): 
```
Adapted from np.moveaxis
- `start`: `Any`
    >No description...
- `end`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.concatenate" class="docs-object-method">&nbsp;</a>
```python
concatenate(self, *others, axis=0): 
```
Concatenates multiple SparseArrays along the specified axis
- `:returns`: `SparseArray`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.broadcast_to" class="docs-object-method">&nbsp;</a>
```python
broadcast_to(self, shape): 
```
Broadcasts self to the given shape.
        Incredibly inefficient implementation but useful in smaller cases.
        Might need to optimize later.
- `shape`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

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

<a id="McUtils.Numputils.Sparse.SparseArray.true_multiply" class="docs-object-method">&nbsp;</a>
```python
true_multiply(self, other): 
```
Multiplies self and other
- `other`: `Any`
    >No description...
- `:returns`: `SparseArray`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.multiply" class="docs-object-method">&nbsp;</a>
```python
multiply(self, other): 
```
Multiplies self and other but allows for broadcasting
- `other`: `SparseArray | np.ndarray | int | float`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.dot" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Numputils.Sparse.SparseArray.tensordot" class="docs-object-method">&nbsp;</a>
```python
tensordot(self, other, axes=2): 
```
Takes the dot product of self and other along the specified axes
- `other`: `Any`
    >No description...
- `axes`: `Iterable[int] | Iterable[Iterable[int]]`
    >the axes to contract along
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.get_caching_status" class="docs-object-method">&nbsp;</a>
```python
get_caching_status(): 
```
A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to specify if caching is on or not
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.enable_caches" class="docs-object-method">&nbsp;</a>
```python
enable_caches(): 
```
A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this on
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.disable_caches" class="docs-object-method">&nbsp;</a>
```python
disable_caches(): 
```
A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this off
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.clear_cache" class="docs-object-method">&nbsp;</a>
```python
clear_cache(): 
```
A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to clear this out.
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Numputils/Sparse/SparseArray.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Numputils/Sparse/SparseArray.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Numputils/Sparse/SparseArray.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Numputils/Sparse/SparseArray.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/Sparse.py?message=Update%20Docs)