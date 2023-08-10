## <a id="McUtils.Numputils.Sparse.ScipySparseArray">ScipySparseArray</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L665)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L665?message=Update%20Docs)]
</div>

Array class that generalize the regular `scipy.sparse.spmatrix`.
Basically acts like a high-dimensional wrapper that manages the _shape_ of a standard `scipy.sparse_matrix`, since that is rigidly 2D.
We always use a combo of an underlying CSR or CSC matrix & COO-like shape operations.







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
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
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L672)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L672?message=Update%20Docs)]
</div>

  - `a`: `Any`
    > 
  - `shape`: `Any`
    > 
  - `layout`: `Any`
    > 
  - `dtype`: `Any`
    > 
  - `initialize`: `Any`
    > 
  - `cache_block_data`: `Any`
    > whether or not
  - `logger`: `Logger`
    > the logger to use for debug purposes


<a id="McUtils.Numputils.Sparse.ScipySparseArray.coo_to_cs" class="docs-object-method">&nbsp;</a> 
```python
coo_to_cs(shape, vals, ij_inds, memmap=False, assume_sorted=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L713)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L713?message=Update%20Docs)]
</div>
Reimplementation of scipy's internal "coo_tocsr" for memory-limited situations
Assumes `ij_inds` are sorted by row then column, which allows vals to be used
directly once indptr is computed
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.to_state" class="docs-object-method">&nbsp;</a> 
```python
to_state(self, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L797)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L797?message=Update%20Docs)]
</div>
Provides just the state that is needed to
serialize the object
  - `serializer`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.from_state" class="docs-object-method">&nbsp;</a> 
```python
from_state(state, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L812)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L812?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.initialize_empty" class="docs-object-method">&nbsp;</a> 
```python
initialize_empty(shape, dtype=None, layout=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L823)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L823?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.construct_sparse_from_val_inds" class="docs-object-method">&nbsp;</a> 
```python
construct_sparse_from_val_inds(a, shape, fmt, cache_block_data=True, logger=None, assume_sorted=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L937)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L937?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.dtype" class="docs-object-method">&nbsp;</a> 
```python
@property
dtype(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1115)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1115?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.diag" class="docs-object-method">&nbsp;</a> 
```python
@property
diag(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1119)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1119?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.from_diagonal_data" class="docs-object-method">&nbsp;</a> 
```python
from_diagonal_data(diags, shape=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1127)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1127?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.asarray" class="docs-object-method">&nbsp;</a> 
```python
asarray(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1143)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1143?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.todense" class="docs-object-method">&nbsp;</a> 
```python
todense(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1145)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1145?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascoo" class="docs-object-method">&nbsp;</a> 
```python
ascoo(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1147)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1147?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascsr" class="docs-object-method">&nbsp;</a> 
```python
ascsr(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1149)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1149?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascsc" class="docs-object-method">&nbsp;</a> 
```python
ascsc(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1151)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1151?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.data" class="docs-object-method">&nbsp;</a> 
```python
@property
data(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1153)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1153?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.format_from_string" class="docs-object-method">&nbsp;</a> 
```python
format_from_string(fmt): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1180)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1180?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.fmt" class="docs-object-method">&nbsp;</a> 
```python
@property
fmt(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1188)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1188?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1196)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1196?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.ndim" class="docs-object-method">&nbsp;</a> 
```python
@property
ndim(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1203)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1203?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.non_zero_count" class="docs-object-method">&nbsp;</a> 
```python
@property
non_zero_count(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1206)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1206?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.get_caching_status" class="docs-object-method">&nbsp;</a> 
```python
get_caching_status(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1213)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1213?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.enable_caches" class="docs-object-method">&nbsp;</a> 
```python
enable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1216)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1216?message=Update%20Docs)]
</div>
A method to be overloaded.
Subclasses may want to cache things for performance, so we
provide a way for them to turn this on
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.disable_caches" class="docs-object-method">&nbsp;</a> 
```python
disable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1226)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1226?message=Update%20Docs)]
</div>
A method to be overloaded.
Subclasses may want to cache things for performance, so we
provide a way for them to turn this off
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.clear_cache" class="docs-object-method">&nbsp;</a> 
```python
clear_cache(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1236)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1236?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.clear_ravel_caches" class="docs-object-method">&nbsp;</a> 
```python
clear_ravel_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1239)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1239?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.set_ravel_cache_size" class="docs-object-method">&nbsp;</a> 
```python
set_ravel_cache_size(size): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1274)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1274?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.find" class="docs-object-method">&nbsp;</a> 
```python
find(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1334)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1334?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_vals" class="docs-object-method">&nbsp;</a> 
```python
@property
block_vals(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1376)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1376?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_inds" class="docs-object-method">&nbsp;</a> 
```python
@property
block_inds(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1388)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1388?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.block_data" class="docs-object-method">&nbsp;</a> 
```python
@property
block_data(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1430)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1430?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.transpose" class="docs-object-method">&nbsp;</a> 
```python
transpose(self, transp): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1445)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1445?message=Update%20Docs)]
</div>
Transposes the array and returns a new one.
Not necessarily a cheap operation.
  - `transp`: `Iterable[int]`
    > the transposition to do
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.reshape" class="docs-object-method">&nbsp;</a> 
```python
reshape(self, shp): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1580)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1580?message=Update%20Docs)]
</div>
Had to make this op not in-place because otherwise got scary errors...
  - `shp`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.squeeze" class="docs-object-method">&nbsp;</a> 
```python
squeeze(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1604)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1604?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.resize" class="docs-object-method">&nbsp;</a> 
```python
resize(self, newsize): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1608)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1608?message=Update%20Docs)]
</div>
Returns a resized version of the tensor
  - `newsize`: `tuple[int]`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate_coo" class="docs-object-method">&nbsp;</a> 
```python
concatenate_coo(self, *others, axis=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1667)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1667?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate_2d" class="docs-object-method">&nbsp;</a> 
```python
concatenate_2d(self, *others, axis=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1742)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1742?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.concatenate" class="docs-object-method">&nbsp;</a> 
```python
concatenate(self, *others, axis=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1757)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1757?message=Update%20Docs)]
</div>
Concatenates multiple arrays along the specified axis
This is relatively inefficient in terms of not tracking indices
throughout
  - `other`: `Any`
    > 
  - `axis`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.broadcast_to" class="docs-object-method">&nbsp;</a> 
```python
broadcast_to(self, shape): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1796)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1796?message=Update%20Docs)]
</div>
Implements broadcast_to using COO-style operations
to be a little bit more efficient
  - `shape`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.T" class="docs-object-method">&nbsp;</a> 
```python
@property
T(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1845)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1845?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__matmul__" class="docs-object-method">&nbsp;</a> 
```python
__matmul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1851)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1851?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.ascs" class="docs-object-method">&nbsp;</a> 
```python
ascs(self, inplace=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1862)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1862?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.dot" class="docs-object-method">&nbsp;</a> 
```python
dot(self, b, reverse=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1868)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1868?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.outer" class="docs-object-method">&nbsp;</a> 
```python
outer(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1903)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1903?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__neg__" class="docs-object-method">&nbsp;</a> 
```python
__neg__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1920)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1920?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__pos__" class="docs-object-method">&nbsp;</a> 
```python
__pos__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1922)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1922?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__add__" class="docs-object-method">&nbsp;</a> 
```python
__add__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1924)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1924?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__iadd__" class="docs-object-method">&nbsp;</a> 
```python
__iadd__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1926)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1926?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__radd__" class="docs-object-method">&nbsp;</a> 
```python
__radd__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1928)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1928?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.plus" class="docs-object-method">&nbsp;</a> 
```python
plus(self, other, inplace=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1930)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1930?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.floopy_flop" class="docs-object-method">&nbsp;</a> 
```python
floopy_flop(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1969)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1969?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__truediv__" class="docs-object-method">&nbsp;</a> 
```python
__truediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1972)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1972?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a> 
```python
__rtruediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1974)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1974?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__rmul__" class="docs-object-method">&nbsp;</a> 
```python
__rmul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1978)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1978?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1980)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1980?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.true_multiply" class="docs-object-method">&nbsp;</a> 
```python
true_multiply(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L1982)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L1982?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.copy" class="docs-object-method">&nbsp;</a> 
```python
copy(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L2013)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L2013?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.savez" class="docs-object-method">&nbsp;</a> 
```python
savez(self, file, compressed=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L2320)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L2320?message=Update%20Docs)]
</div>
Saves a SparseArray to a file (must have the npz extension)
  - `file`: `Any`
    > 
  - `compressed`: `Any`
    > 
  - `:returns`: `str`
    > t
h
e
 
s
a
v
e
d
 
f
i
l
e


<a id="McUtils.Numputils.Sparse.ScipySparseArray.loadz" class="docs-object-method">&nbsp;</a> 
```python
loadz(file): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L2359)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L2359?message=Update%20Docs)]
</div>
Loads a SparseArray from an npz file
  - `file`: `Any`
    > 
  - `:returns`: `SparseArray`
    >


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L2375)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L2375?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__setitem__" class="docs-object-method">&nbsp;</a> 
```python
__setitem__(self, item, val): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L2377)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L2377?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__delitem__" class="docs-object-method">&nbsp;</a> 
```python
__delitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L2379)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L2379?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.ScipySparseArray.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/ScipySparseArray.py#L2382)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/ScipySparseArray.py#L2382?message=Update%20Docs)]
</div>
 </div>
</div>












---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Numputils/Sparse/ScipySparseArray.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Numputils/Sparse/ScipySparseArray.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Numputils/Sparse/ScipySparseArray.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Numputils/Sparse/ScipySparseArray.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L665?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>