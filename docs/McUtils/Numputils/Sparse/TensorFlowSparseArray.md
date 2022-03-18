## <a id="McUtils.Numputils.Sparse.TensorFlowSparseArray">TensorFlowSparseArray</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2349)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2349?message=Update%20Docs)]
</div>

Provides a SparseArray implementation that uses TensorFlow as the backend

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, data, dtype=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2353)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2353?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

Provides the shape of the sparse array
- `:returns`: `tuple[int]`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.to_state" class="docs-object-method">&nbsp;</a> 
```python
to_state(self, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2416)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2416?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2427)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2427?message=Update%20Docs)]
</div>

Loads from the stored state
- `serializer`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.empty" class="docs-object-method">&nbsp;</a> 
```python
empty(shape, dtype=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2438)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2438?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

Returns the row and column indices and vector of
        values that the sparse array is storing
- `:returns`: `Tuple[np.ndarray, Iterable[np.ndarray]]`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.transpose" class="docs-object-method">&nbsp;</a> 
```python
transpose(self, axes): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2463)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2463?message=Update%20Docs)]
</div>

Returns a transposed version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.ascoo" class="docs-object-method">&nbsp;</a> 
```python
ascoo(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2475)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2475?message=Update%20Docs)]
</div>

Converts the tensor into a scipy COO matrix...
- `:returns`: `sp.coo_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.ascsr" class="docs-object-method">&nbsp;</a> 
```python
ascsr(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2482)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2482?message=Update%20Docs)]
</div>

Converts the tensor into a scipy COO matrix...
- `:returns`: `sp.coo_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.reshape" class="docs-object-method">&nbsp;</a> 
```python
reshape(self, newshape): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2489)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2489?message=Update%20Docs)]
</div>

Returns a reshaped version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__truediv__" class="docs-object-method">&nbsp;</a> 
```python
__truediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2501)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2501?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a> 
```python
__rtruediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2504)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2504?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__rmul__" class="docs-object-method">&nbsp;</a> 
```python
__rmul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2507)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2507?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2510)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2510?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.true_multiply" class="docs-object-method">&nbsp;</a> 
```python
true_multiply(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2513)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2513?message=Update%20Docs)]
</div>

Multiplies self and other
- `other`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.TensorFlowSparseArray.dot" class="docs-object-method">&nbsp;</a> 
```python
dot(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L2567)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2567?message=Update%20Docs)]
</div>

Takes a regular dot product of self and other
- `other`: `Any`
    >No description...
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Numputils/Sparse/TensorFlowSparseArray.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L2349?message=Update%20Docs)