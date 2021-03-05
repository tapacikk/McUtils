## <a id="McUtils.Numputils.Sparse.SparseArray">SparseArray</a>
Represents a generic sparse array format
which can be subclassed to provide a concrete implementation

### Properties and Methods
```python
backends: NoneType
get_backends: method
from_data: method
from_diag: method
from_diagonal_data: method
from_state: method
empty: method
get_caching_status: method
enable_caches: method
disable_caches: method
clear_cache: method
```
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

<a id="McUtils.Numputils.Sparse.SparseArray.block_data" class="docs-object-method">&nbsp;</a>
```python
@property
block_data(self): 
```
Returns the row and column indices and vector of
        values that the sparse array is storing
- `shape`: `Any`
    >No description...
- `dtype`: `Any`
    >No description...
- `kw`: `Any`
    >No description...
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
Multiplies self and other
- `other`: `Any`
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

<a id="McUtils.Numputils.Sparse.SparseArray.concatenate" class="docs-object-method">&nbsp;</a>
```python
concatenate(self, other, axis=0): 
```
Concatenates two SparseArrays along the specified axis
- `:returns`: `SparseArray`
    >No description...

### Examples


