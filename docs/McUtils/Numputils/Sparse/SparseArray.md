## <a id="McUtils.Numputils.Sparse.SparseArray">SparseArray</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse.py#L23)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L23?message=Update%20Docs)]
</div>

Represents a generic sparse array format
which can be subclassed to provide a concrete implementation







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 ```python
backends: NoneType
cacheing_manager: cacheing_manager
initializer_list: initializer_list
```
<a id="McUtils.Numputils.Sparse.SparseArray.get_backends" class="docs-object-method">&nbsp;</a> 
```python
get_backends(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L29)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L29?message=Update%20Docs)]
</div>
Provides the set of backends to try by default
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.from_data" class="docs-object-method">&nbsp;</a> 
```python
from_data(data, shape=None, dtype=None, target_backend=None, constructor=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L41)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L41?message=Update%20Docs)]
</div>
A wrapper so that we can dispatch to the best
sparse backend we've got defined.
Can be monkey patched.
  - `data`: `Any`
    > 
  - `kwargs`: `Any`
    > 
  - `:returns`: `SparseArray`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.from_diag" class="docs-object-method">&nbsp;</a> 
```python
from_diag(data, shape=None, dtype=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L82)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L82?message=Update%20Docs)]
</div>
A wrapper so that we can dispatch to the best
sparse backend we've got defined.
Can be monkey patched.
  - `data`: `Any`
    > 
  - `kwargs`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.from_diagonal_data" class="docs-object-method">&nbsp;</a> 
```python
from_diagonal_data(diags, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L100)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L100?message=Update%20Docs)]
</div>
Constructs a sparse tensor from diagonal elements
  - `diags`: `Any`
    > 
  - `kw`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L115)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L115?message=Update%20Docs)]
</div>
Provides the shape of the sparse array
  - `:returns`: `tuple[int]`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.ndim" class="docs-object-method">&nbsp;</a> 
```python
@property
ndim(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L124)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L124?message=Update%20Docs)]
</div>
Provides the number of dimensions in the array
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.to_state" class="docs-object-method">&nbsp;</a> 
```python
to_state(self, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L132)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L132?message=Update%20Docs)]
</div>
Provides just the state that is needed to
serialize the object
  - `serializer`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.from_state" class="docs-object-method">&nbsp;</a> 
```python
from_state(state, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L143)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L143?message=Update%20Docs)]
</div>
Loads from the stored state
  - `serializer`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.empty" class="docs-object-method">&nbsp;</a> 
```python
empty(shape, dtype=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L155)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L155?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.SparseArray.initialize_empty" class="docs-object-method">&nbsp;</a> 
```python
initialize_empty(shp, shape=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L162)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L162?message=Update%20Docs)]
</div>
Returns an empty SparseArray with the appropriate shape and dtype
  - `shape`: `Any`
    > 
  - `dtype`: `Any`
    > 
  - `kw`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.block_data" class="docs-object-method">&nbsp;</a> 
```python
@property
block_data(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L178)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L178?message=Update%20Docs)]
</div>
Returns the vector of values and corresponding indices
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.block_inds" class="docs-object-method">&nbsp;</a> 
```python
@property
block_inds(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L188)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L188?message=Update%20Docs)]
</div>
Returns indices for the stored values
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.transpose" class="docs-object-method">&nbsp;</a> 
```python
transpose(self, axes): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L198)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L198?message=Update%20Docs)]
</div>
Returns a transposed version of the tensor
  - `axes`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.ascoo" class="docs-object-method">&nbsp;</a> 
```python
ascoo(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L209)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L209?message=Update%20Docs)]
</div>
Converts the tensor into a scipy COO matrix...
  - `:returns`: `sp.coo_matrix`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.ascsr" class="docs-object-method">&nbsp;</a> 
```python
ascsr(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L218)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L218?message=Update%20Docs)]
</div>
Converts the tensor into a scipy CSR matrix...
  - `:returns`: `sp.csr_matrix`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.asarray" class="docs-object-method">&nbsp;</a> 
```python
asarray(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L226)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L226?message=Update%20Docs)]
</div>
Converts the tensor into a dense np.ndarray
  - `:returns`: `np.ndarray`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.reshape" class="docs-object-method">&nbsp;</a> 
```python
reshape(self, newshape): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L234)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L234?message=Update%20Docs)]
</div>
Returns a reshaped version of the tensor
  - `axes`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.resize" class="docs-object-method">&nbsp;</a> 
```python
resize(self, newsize): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L244)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L244?message=Update%20Docs)]
</div>
Returns a resized version of the tensor
  - `axes`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.expand_dims" class="docs-object-method">&nbsp;</a> 
```python
expand_dims(self, axis): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L255)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L255?message=Update%20Docs)]
</div>
adapted from np.expand_dims
  - `axis`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.moveaxis" class="docs-object-method">&nbsp;</a> 
```python
moveaxis(self, start, end): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L276)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L276?message=Update%20Docs)]
</div>
Adapted from np.moveaxis
  - `start`: `Any`
    > 
  - `end`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.concatenate" class="docs-object-method">&nbsp;</a> 
```python
concatenate(self, *others, axis=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L305)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L305?message=Update%20Docs)]
</div>
Concatenates multiple SparseArrays along the specified axis
  - `:returns`: `SparseArray`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.broadcast_to" class="docs-object-method">&nbsp;</a> 
```python
broadcast_to(self, shape): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L314)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L314?message=Update%20Docs)]
</div>
Broadcasts self to the given shape.
Incredibly inefficient implementation but useful in smaller cases.
Might need to optimize later.
  - `shape`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.__truediv__" class="docs-object-method">&nbsp;</a> 
```python
__truediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L344)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L344?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.SparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a> 
```python
__rtruediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L346)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L346?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.SparseArray.__rmul__" class="docs-object-method">&nbsp;</a> 
```python
__rmul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L348)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L348?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.SparseArray.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L350)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L350?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.SparseArray.true_multiply" class="docs-object-method">&nbsp;</a> 
```python
true_multiply(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L384)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L384?message=Update%20Docs)]
</div>
Multiplies self and other
  - `other`: `Any`
    > 
  - `:returns`: `SparseArray`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.multiply" class="docs-object-method">&nbsp;</a> 
```python
multiply(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L394)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L394?message=Update%20Docs)]
</div>
Multiplies self and other but allows for broadcasting
  - `other`: `SparseArray | np.ndarray | int | float`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.dot" class="docs-object-method">&nbsp;</a> 
```python
dot(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L409)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L409?message=Update%20Docs)]
</div>
Takes a regular dot product of self and other
  - `other`: `Any`
    > 
  - `axes`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.tensordot" class="docs-object-method">&nbsp;</a> 
```python
tensordot(self, other, axes=2): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L422)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L422?message=Update%20Docs)]
</div>
Takes the dot product of self and other along the specified axes
  - `other`: `Any`
    > 
  - `axes`: `Iterable[int] | Iterable[Iterable[int]]`
    > the axes to contract along
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.cache_options" class="docs-object-method">&nbsp;</a> 
```python
cache_options(enabled=True, clear=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L562)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L562?message=Update%20Docs)]
</div>


<a id="McUtils.Numputils.Sparse.SparseArray.get_caching_status" class="docs-object-method">&nbsp;</a> 
```python
get_caching_status(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L565)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L565?message=Update%20Docs)]
</div>
A method to be overloaded.
Subclasses may want to cache things for performance, so we
provide a way for them to specify if caching is on or not
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.enable_caches" class="docs-object-method">&nbsp;</a> 
```python
enable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L574)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L574?message=Update%20Docs)]
</div>
A method to be overloaded.
Subclasses may want to cache things for performance, so we
provide a way for them to turn this on
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.disable_caches" class="docs-object-method">&nbsp;</a> 
```python
disable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L583)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L583?message=Update%20Docs)]
</div>
A method to be overloaded.
Subclasses may want to cache things for performance, so we
provide a way for them to turn this off
  - `:returns`: `_`
    >


<a id="McUtils.Numputils.Sparse.SparseArray.clear_cache" class="docs-object-method">&nbsp;</a> 
```python
clear_cache(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Numputils/Sparse/SparseArray.py#L592)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse/SparseArray.py#L592?message=Update%20Docs)]
</div>
A method to be overloaded.
Subclasses may want to cache things for performance, so we
provide a way for them to clear this out.
  - `:returns`: `_`
    >
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Numputils/Sparse/SparseArray.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Numputils/Sparse/SparseArray.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Numputils/Sparse/SparseArray.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Numputils/Sparse/SparseArray.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Numputils/Sparse.py#L23?message=Update%20Docs)   
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