## <a id="McUtils.Numputils.Sparse.ScipySparseArray">ScipySparseArray</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L649)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L649?message=Update%20Docs)]
</div>

Array class that generalize the regular `scipy.sparse.spmatrix`.
Basically acts like a high-dimensional wrapper that manages the _shape_ of a standard `scipy.sparse_matrix`, since that is rigidly 2D.
We always use a combo of an underlying CSR or CSC matrix & COO-like shape operations.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
formats_map: dict
default_cache_size: int
caching_enabled: bool
```
<a id="McUtils.Numputils.Sparse.ScipySparseArray.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, a, shape=None, layout=None, dtype=None, initialize=True, cache_block_data=None, logger=None, init_kwargs=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L656)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L656?message=Update%20Docs)]
</div>


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

<a id="McUtils.Numputils.Sparse.ScipySparseArray.coo_to_cs" class="docs-object-method">&nbsp;</a> 
```python
coo_to_cs(shape, vals, ij_inds, memmap=False, assume_sorted=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L697)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L697?message=Update%20Docs)]
</div>

Reimplementation of scipy's internal "coo_tocsr" for memory-limited situations
        Assumes `ij_inds` are sorted by row then column, which allows vals to be used
        directly once indptr is computed
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.to_state" class="docs-object-method">&nbsp;</a> 
```python
to_state(self, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L781)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L781?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L796)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L796?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.initialize_empty" class="docs-object-method">&nbsp;</a> 
```python
initialize_empty(shape, dtype=None, layout=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L807)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L807?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.construct_sparse_from_val_inds" class="docs-object-method">&nbsp;</a> 
```python
construct_sparse_from_val_inds(a, shape, fmt, cache_block_data=True, logger=None, assume_sorted=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L921)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L921?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.dtype" class="docs-object-method">&nbsp;</a> 
```python
@property
dtype(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.diag" class="docs-object-method">&nbsp;</a> 
```python
@property
diag(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.from_diagonal_data" class="docs-object-method">&nbsp;</a> 
```python
from_diagonal_data(diags, shape=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1111)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1111?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.asarray" class="docs-object-method">&nbsp;</a> 
```python
asarray(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1127)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1127?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.todense" class="docs-object-method">&nbsp;</a> 
```python
todense(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1129)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1129?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascoo" class="docs-object-method">&nbsp;</a> 
```python
ascoo(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1131)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1131?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascsr" class="docs-object-method">&nbsp;</a> 
```python
ascsr(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1133)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1133?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascsc" class="docs-object-method">&nbsp;</a> 
```python
ascsc(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1135)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1135?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.data" class="docs-object-method">&nbsp;</a> 
```python
@property
data(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.format_from_string" class="docs-object-method">&nbsp;</a> 
```python
format_from_string(fmt): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1164)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1164?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.fmt" class="docs-object-method">&nbsp;</a> 
```python
@property
fmt(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ndim" class="docs-object-method">&nbsp;</a> 
```python
@property
ndim(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.non_zero_count" class="docs-object-method">&nbsp;</a> 
```python
@property
non_zero_count(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.get_caching_status" class="docs-object-method">&nbsp;</a> 
```python
get_caching_status(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1197)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1197?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.enable_caches" class="docs-object-method">&nbsp;</a> 
```python
enable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1200)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1200?message=Update%20Docs)]
</div>

A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this on
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.disable_caches" class="docs-object-method">&nbsp;</a> 
```python
disable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1210)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1210?message=Update%20Docs)]
</div>

A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this off
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.clear_cache" class="docs-object-method">&nbsp;</a> 
```python
clear_cache(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1220)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1220?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.clear_ravel_caches" class="docs-object-method">&nbsp;</a> 
```python
clear_ravel_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1223)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1223?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.set_ravel_cache_size" class="docs-object-method">&nbsp;</a> 
```python
set_ravel_cache_size(size): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1258)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1258?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.find" class="docs-object-method">&nbsp;</a> 
```python
find(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1318)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1318?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_vals" class="docs-object-method">&nbsp;</a> 
```python
@property
block_vals(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_inds" class="docs-object-method">&nbsp;</a> 
```python
@property
block_inds(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_data" class="docs-object-method">&nbsp;</a> 
```python
@property
block_data(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.transpose" class="docs-object-method">&nbsp;</a> 
```python
transpose(self, transp): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1429)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1429?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1564)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1564?message=Update%20Docs)]
</div>

Had to make this op not in-place because otherwise got scary errors...
- `shp`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.squeeze" class="docs-object-method">&nbsp;</a> 
```python
squeeze(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1588)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1588?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.resize" class="docs-object-method">&nbsp;</a> 
```python
resize(self, newsize): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1592)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1592?message=Update%20Docs)]
</div>

Returns a resized version of the tensor
- `newsize`: `tuple[int]`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate_coo" class="docs-object-method">&nbsp;</a> 
```python
concatenate_coo(self, *others, axis=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1651)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1651?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate_2d" class="docs-object-method">&nbsp;</a> 
```python
concatenate_2d(self, *others, axis=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1723)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1723?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate" class="docs-object-method">&nbsp;</a> 
```python
concatenate(self, *others, axis=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1736)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1736?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1775)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1775?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__matmul__" class="docs-object-method">&nbsp;</a> 
```python
__matmul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1830)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1830?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascs" class="docs-object-method">&nbsp;</a> 
```python
ascs(self, inplace=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1841)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1841?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.dot" class="docs-object-method">&nbsp;</a> 
```python
dot(self, b, reverse=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1847)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1847?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__neg__" class="docs-object-method">&nbsp;</a> 
```python
__neg__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1882)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1882?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__pos__" class="docs-object-method">&nbsp;</a> 
```python
__pos__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1884)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1884?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__add__" class="docs-object-method">&nbsp;</a> 
```python
__add__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1886)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1886?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__iadd__" class="docs-object-method">&nbsp;</a> 
```python
__iadd__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1888)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1888?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__radd__" class="docs-object-method">&nbsp;</a> 
```python
__radd__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1890)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1890?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.plus" class="docs-object-method">&nbsp;</a> 
```python
plus(self, other, inplace=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1892)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1892?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.floopy_flop" class="docs-object-method">&nbsp;</a> 
```python
floopy_flop(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1931)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1931?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__truediv__" class="docs-object-method">&nbsp;</a> 
```python
__truediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1934)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1934?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a> 
```python
__rtruediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1936)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1936?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__rmul__" class="docs-object-method">&nbsp;</a> 
```python
__rmul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1940)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1940?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1942)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1942?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.true_multiply" class="docs-object-method">&nbsp;</a> 
```python
true_multiply(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1944)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1944?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.copy" class="docs-object-method">&nbsp;</a> 
```python
copy(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L1975)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L1975?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.savez" class="docs-object-method">&nbsp;</a> 
```python
savez(self, file, compressed=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L2275)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L2275?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L2314)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L2314?message=Update%20Docs)]
</div>

Loads a SparseArray from an npz file
- `file`: `Any`
    >No description...
- `:returns`: `SparseArray`
    >No description...

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L2330)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L2330?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__setitem__" class="docs-object-method">&nbsp;</a> 
```python
__setitem__(self, item, val): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L2332)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L2332?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__delitem__" class="docs-object-method">&nbsp;</a> 
```python
__delitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L2334)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L2334?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.ScipySparseArray.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L2337)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L2337?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Numputils/Sparse/ScipySparseArray.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Numputils/Sparse/ScipySparseArray.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Numputils/Sparse/ScipySparseArray.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Numputils/Sparse/ScipySparseArray.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L649?message=Update%20Docs)