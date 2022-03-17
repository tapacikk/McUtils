## <a id="McUtils.Numputils.Sparse.SparseArray">SparseArray</a>
Represents a generic sparse array format
which can be subclassed to provide a concrete implementation

### Properties and Methods
```python
backends: NoneType
cacheing_manager: type
initializer_list: type
```
<a id="McUtils.Numputils.Sparse.SparseArray.get_backends" class="docs-object-method">&nbsp;</a> 
```python
get_backends(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L29)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L29?message=Update%20Docs)]
</div>

Provides the set of backends to try by default
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.from_data" class="docs-object-method">&nbsp;</a> 
```python
from_data(data, shape=None, dtype=None, target_backend=None, constructor=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L41)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L41?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L82)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L82?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L100)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L100?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

Provides the shape of the sparse array
- `:returns`: `tuple[int]`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.ndim" class="docs-object-method">&nbsp;</a> 
```python
@property
ndim(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

Provides the number of dimensions in the array
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.to_state" class="docs-object-method">&nbsp;</a> 
```python
to_state(self, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L132)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L132?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L143)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L143?message=Update%20Docs)]
</div>

Loads from the stored state
- `serializer`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.empty" class="docs-object-method">&nbsp;</a> 
```python
empty(shape, dtype=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L155)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L155?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.SparseArray.initialize_empty" class="docs-object-method">&nbsp;</a> 
```python
initialize_empty(shp, shape=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L162)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L162?message=Update%20Docs)]
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

<a id="McUtils.Numputils.Sparse.SparseArray.block_data" class="docs-object-method">&nbsp;</a> 
```python
@property
block_data(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

Returns the vector of values and corresponding indices
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.block_inds" class="docs-object-method">&nbsp;</a> 
```python
@property
block_inds(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L?message=Update%20Docs)]
</div>

Returns indices for the stored values
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.transpose" class="docs-object-method">&nbsp;</a> 
```python
transpose(self, axes): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L198)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L198?message=Update%20Docs)]
</div>

Returns a transposed version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.ascoo" class="docs-object-method">&nbsp;</a> 
```python
ascoo(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L209)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L209?message=Update%20Docs)]
</div>

Converts the tensor into a scipy COO matrix...
- `:returns`: `sp.coo_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.ascsr" class="docs-object-method">&nbsp;</a> 
```python
ascsr(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L218)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L218?message=Update%20Docs)]
</div>

Converts the tensor into a scipy CSR matrix...
- `:returns`: `sp.csr_matrix`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.asarray" class="docs-object-method">&nbsp;</a> 
```python
asarray(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L226)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L226?message=Update%20Docs)]
</div>

Converts the tensor into a dense np.ndarray
- `:returns`: `np.ndarray`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.reshape" class="docs-object-method">&nbsp;</a> 
```python
reshape(self, newshape): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L234)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L234?message=Update%20Docs)]
</div>

Returns a reshaped version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.resize" class="docs-object-method">&nbsp;</a> 
```python
resize(self, newsize): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L244)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L244?message=Update%20Docs)]
</div>

Returns a resized version of the tensor
- `axes`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.expand_dims" class="docs-object-method">&nbsp;</a> 
```python
expand_dims(self, axis): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L255?message=Update%20Docs)]
</div>

adapted from np.expand_dims
- `axis`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.moveaxis" class="docs-object-method">&nbsp;</a> 
```python
moveaxis(self, start, end): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L276)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L276?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L305)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L305?message=Update%20Docs)]
</div>

Concatenates multiple SparseArrays along the specified axis
- `:returns`: `SparseArray`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.broadcast_to" class="docs-object-method">&nbsp;</a> 
```python
broadcast_to(self, shape): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L314)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L314?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L344)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L344?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.SparseArray.__rtruediv__" class="docs-object-method">&nbsp;</a> 
```python
__rtruediv__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L346)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L346?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.SparseArray.__rmul__" class="docs-object-method">&nbsp;</a> 
```python
__rmul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L348)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L348?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.SparseArray.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L350)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L350?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.SparseArray.true_multiply" class="docs-object-method">&nbsp;</a> 
```python
true_multiply(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L384)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L384?message=Update%20Docs)]
</div>

Multiplies self and other
- `other`: `Any`
    >No description...
- `:returns`: `SparseArray`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.multiply" class="docs-object-method">&nbsp;</a> 
```python
multiply(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L394)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L394?message=Update%20Docs)]
</div>

Multiplies self and other but allows for broadcasting
- `other`: `SparseArray | np.ndarray | int | float`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.dot" class="docs-object-method">&nbsp;</a> 
```python
dot(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L409)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L409?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L422)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L422?message=Update%20Docs)]
</div>

Takes the dot product of self and other along the specified axes
- `other`: `Any`
    >No description...
- `axes`: `Iterable[int] | Iterable[Iterable[int]]`
    >the axes to contract along
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.cache_options" class="docs-object-method">&nbsp;</a> 
```python
cache_options(enabled=True, clear=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L562)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L562?message=Update%20Docs)]
</div>

<a id="McUtils.Numputils.Sparse.SparseArray.get_caching_status" class="docs-object-method">&nbsp;</a> 
```python
get_caching_status(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L565)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L565?message=Update%20Docs)]
</div>

A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to specify if caching is on or not
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.enable_caches" class="docs-object-method">&nbsp;</a> 
```python
enable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L574)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L574?message=Update%20Docs)]
</div>

A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this on
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.disable_caches" class="docs-object-method">&nbsp;</a> 
```python
disable_caches(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L583)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L583?message=Update%20Docs)]
</div>

A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to turn this off
- `:returns`: `_`
    >No description...

<a id="McUtils.Numputils.Sparse.SparseArray.clear_cache" class="docs-object-method">&nbsp;</a> 
```python
clear_cache(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Numputils/Sparse.py#L592)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Numputils/Sparse.py#L592?message=Update%20Docs)]
</div>

A method to be overloaded.
        Subclasses may want to cache things for performance, so we
        provide a way for them to clear this out.
- `:returns`: `_`
    >No description...




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [SparseArray](#SparseArray)
- [Sparse](#Sparse)
- [SparseConstructor](#SparseConstructor)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
#### <a class="collapse-link" data-toggle="collapse" href="#test-setup">Setup</a> <a class="float-right" data-toggle="collapse" href="#test-setup"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="test-setup" markdown="1">

Before we can run our examples we should get a bit of setup out of the way.
Since these examples were harvested from the unit tests not all pieces
will be necessary for all situations.
```python
from Peeves.TestUtils import *
from Peeves import BlockProfiler
from McUtils.Numputils import *
from McUtils.Zachary import FiniteDifferenceDerivative
from unittest import TestCase
import numpy as np, functools as ft
```

All tests are wrapped in a test class
```python
class NumputilsTests(TestCase):
    problem_coords = np.array([
                                  [-1.86403557e-17, -7.60465240e-02,  4.62443228e-02],
                                  [ 6.70904773e-17, -7.60465240e-02, -9.53755677e-01],
                                  [ 9.29682337e-01,  2.92315732e-01,  4.62443228e-02],
                                  [ 2.46519033e-32, -1.38777878e-17,  2.25076602e-01],
                                  [-1.97215226e-31,  1.43714410e+00, -9.00306410e-01],
                                  [-1.75999392e-16, -1.43714410e+00, -9.00306410e-01]
    ])
```

 </div>
</div>

#### <a name="SparseArray">SparseArray</a>
```python
    def test_SparseArray(self):
        array = SparseArray([
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ],
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ],
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 1]
            ],
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]
            ]
        ])

        self.assertEquals(array.shape, (4, 3, 3))
        tp = array.transpose((1, 0, 2))
        self.assertEquals(tp.shape, (3, 4, 3))
        self.assertLess(np.linalg.norm((tp.asarray()-array.asarray().transpose((1, 0, 2))).flatten()), 1e-8)
        self.assertEquals(array[2, :, 2].shape, (3,))
        td = array.tensordot(array, axes=[1, 1])
        self.assertEquals(td.shape, (4, 3, 4, 3))
        self.assertEquals(array.tensordot(array, axes=[[1, 2], [1, 2]]).shape, (4, 4))
```
#### <a name="Sparse">Sparse</a>
```python
    def test_Sparse(self):

        shape = (1000, 100, 50)

        n_els = 100
        np.random.seed(1)
        inds = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals = np.random.rand(len(inds))
        inds = inds.T

        # `from_data` for backend flexibility
        array = SparseArray.from_data(
            (
                vals,
                inds
            ),
            shape=shape
        )


        self.assertEquals(array.shape, shape)
        block_vals, block_inds = array.block_data
        self.assertEquals(len(block_vals), len(vals))
        self.assertEquals(np.sort(block_vals).tolist(), np.sort(vals).tolist())
        for i in range(len(shape)):
            self.assertEquals(np.sort(block_inds[i]).tolist(), np.sort(inds[i]).tolist())

        woof = array[:, 1, 1] #type: SparseArray
        self.assertIs(type(woof), type(array))
        self.assertEquals(woof.shape, (shape[0],))
        block_vals, block_inds = woof.block_data
        filt_pos = np.where(np.logical_and(inds[1] == 1, inds[2] == 1))
        if len(filt_pos) > 0:
            self.assertEquals(
                np.sort(block_vals).tolist(),
                np.sort(vals[filt_pos]).tolist()
            )

        # with BlockProfiler('Sparse sampling', print_res=True):
        #     new_woof = array[:, 1, 1]  # type: SparseArray

        shape = (28, 3003)

        n_els = 10000
        np.random.seed(1)
        inds = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals = np.random.rand(len(inds))
        inds = inds.T

        # `from_data` for backend flexibility
        array = SparseArray.from_data(
            (
                vals,
                inds
            ),
            shape=shape
        )

        woof = array[0, :]  # type: SparseArray
        self.assertIs(type(woof), type(array))
        self.assertEquals(woof.shape, (shape[1],))
        block_vals, block_inds = woof.block_data
        filt_pos = np.where(inds[0] == 0)
        if len(filt_pos) > 0:
            self.assertEquals(
                np.sort(block_vals).tolist(),
                np.sort(vals[filt_pos]).tolist()
            )

        woof = array[(0, 2), :]  # type: SparseArray
        self.assertEquals(woof.shape, (2, shape[1]))
        block_vals, block_inds = woof.block_data
        filt_pos = np.where(np.logical_or(inds[0] == 0, inds[0] == 2))
        if len(filt_pos) > 0:
            self.assertEquals(
                np.sort(block_vals).tolist(),
                np.sort(vals[filt_pos]).tolist()
            )

            self.assertEquals(
                block_vals[:10].tolist(),
                [0.26762682146970584, 0.3742446513095977, 0.11369722324344822, 0.4860704109280778,
                 0.09299008335958303, 0.11229999691948178, 0.0005348158154161453, 0.7711636892670307, 0.6573053253883241, 0.39084691369185387]

            )

        n_els = 1000
        inds_2 = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals_2 = np.random.rand(len(inds_2))
        inds_2 = inds_2.T

        # `from_data` for backend flexibility
        array_2 = SparseArray.from_data(
            (
                vals_2,
                inds_2
            ),
            shape=shape
        )

        meh = array.dot(array_2.transpose((1, 0)))
        self.assertTrue(
            np.allclose(
                meh.asarray(),
                np.dot(
                    array.asarray(),
                    array_2.asarray().T
                ),
                3
            )
        )

        n_els = 1000
        inds_3 = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals_3 = np.random.rand(len(inds_3))
        inds_3 = inds_3.T

        # `from_data` for backend flexibility
        array_3 = SparseArray.from_data(
            (
                vals_3,
                inds_3
            ),
            shape=shape
        )

        new2 = array_2.concatenate(array_3)
        meh = np.concatenate([array_2.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )



        new2 = array_2.concatenate(array_3, array_2)
        meh = np.concatenate([array_2.asarray(), array_3.asarray(), array_2.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat many failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new3 = array_2.concatenate(array_2, array_3, axis=1)
        meh = np.concatenate([array_2.asarray(), array_2.asarray(), array_3.asarray()], axis=1)
        self.assertEquals(new3.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new3.asarray(),
                meh
            ),
            msg="concat along 1 failed: (ref) {} vs {}".format(
                meh,
                new3.asarray()
            )
        )

        new_shape = [1, shape[1]]

        n_els = 1000
        inds_3 = np.unique(np.array([np.random.choice(x, n_els) for x in new_shape]).T, axis=0)
        vals_3 = np.random.rand(len(inds_3))
        inds_3 = inds_3.T

        # `from_data` for backend flexibility
        array_3 = SparseArray.from_data(
            (
                vals_3,
                inds_3
            ),
            shape=new_shape
        )

        new2 = array_3.concatenate(array_2)
        meh = np.concatenate([array_3.asarray(), array_2.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new2 = array_2.concatenate(array_3)
        meh = np.concatenate([array_2.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new2 = array_2.concatenate(array_3, array_2)
        meh = np.concatenate([array_2.asarray(), array_3.asarray(), array_2.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat many failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        # new3 = array_2.concatenate(array_2, array_3, axis=1)
        # meh = np.concatenate([array_2.asarray(), array_2.asarray(), array_3.asarray()], axis=1)
        # self.assertEquals(new3.shape, meh.shape)
        # self.assertTrue(
        #     np.allclose(
        #         new3.asarray(),
        #         meh
        #     ),
        #     msg="concat along 1 failed: (ref) {} vs {}".format(
        #         meh,
        #         new3.asarray()
        #     )
        # )

        array_3 = array_3[:, :2500].reshape((1, 2500))

        array_3 = array_3.reshape((
                array_3.shape[1] // 2,
                2
        ))

        new2 = array_3.concatenate(array_3)
        meh = np.concatenate([array_3.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new2 = array_3.concatenate(array_3, axis=1)
        meh = np.concatenate([array_3.asarray(), array_3.asarray()], axis=1)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new_shape = [shape[1]]

        n_els = 1000
        inds_3 = np.unique(np.array([np.random.choice(x, n_els) for x in new_shape]).T, axis=0)
        vals_3 = np.random.rand(len(inds_3))
        inds_3 = inds_3.T

        array_3 = SparseArray.from_data(
            (
                vals_3,
                inds_3
            ),
            shape=new_shape
        )

        new2 = array_3.concatenate(array_3)
        meh = np.concatenate([array_3.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        wtf_array1 = SparseArray.from_data(
            (
                [-0.00045906, -0.00045906, -0.00045906, -0.00045906, -0.00045906,
                 -0.00045906],
                (
                    (0, 24, 51, 78, 109, 140),
                )
            ),
            shape = (155,)

        )

        wtf_array2 = SparseArray.from_data(
            (
                [-0.00045906, -0.00045906, -0.00045906, -0.00045906],
                ([ 16,  53,  88, 123],)
            ),
            shape=(155,)
        )

        new2 = wtf_array1.concatenate(wtf_array2)
        meh = np.concatenate([
            wtf_array1.asarray(),
            wtf_array2.asarray()
        ])

        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )
```
#### <a name="SparseConstructor">SparseConstructor</a>
```python
    def test_SparseConstructor(self):

        shape = (1000, 100, 50)

        n_els = 100
        np.random.seed(1)
        inds = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals = np.random.rand(len(inds))
        inds = inds.T

        # `from_data` for backend flexibility
        array = SparseArray.from_data(
            (
                vals,
                inds
            ),
            shape=shape
        )

        self.assertEquals(array.shape, shape)
        block_vals, block_inds = array.block_data
        self.assertEquals(len(block_vals), len(vals))
        self.assertEquals(np.sort(block_vals).tolist(), np.sort(vals).tolist())
        for i in range(len(shape)):
            self.assertEquals(np.sort(block_inds[i]).tolist(), np.sort(inds[i]).tolist())

        woof = array[:, 1, 1]  # type: SparseArray
        self.assertIs(type(woof), type(array))
        self.assertEquals(woof.shape, (shape[0],))
        block_vals, block_inds = woof.block_data
        filt_pos = np.where(np.logical_and(inds[1] == 1, inds[2] == 1))
        if len(filt_pos) > 0:
            self.assertEquals(
                np.sort(block_vals).tolist(),
                np.sort(vals[filt_pos]).tolist()
            )

        # with BlockProfiler('Sparse sampling', print_res=True):
        #     new_woof = array[:, 1, 1]  # type: SparseArray

        shape = (28, 3003)

        n_els = 10000
        np.random.seed(1)
        inds = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals = np.random.rand(len(inds))
        inds = inds.T

        # `from_data` for backend flexibility
        array = SparseArray.from_data(
            (
                vals,
                inds
            ),
            shape=shape
        )

        woof = array[0, :]  # type: SparseArray
        self.assertIs(type(woof), type(array))
        self.assertEquals(woof.shape, (shape[1],))
        block_vals, block_inds = woof.block_data
        filt_pos = np.where(inds[0] == 0)
        if len(filt_pos) > 0:
            self.assertEquals(
                np.sort(block_vals).tolist(),
                np.sort(vals[filt_pos]).tolist()
            )

        woof = array[(0, 2), :]  # type: SparseArray
        self.assertEquals(woof.shape, (2, shape[1]))
        block_vals, block_inds = woof.block_data
        filt_pos = np.where(np.logical_or(inds[0] == 0, inds[0] == 2))
        if len(filt_pos) > 0:
            self.assertEquals(
                np.sort(block_vals).tolist(),
                np.sort(vals[filt_pos]).tolist()
            )

            self.assertEquals(
                block_vals[:10].tolist(),
                [0.26762682146970584, 0.3742446513095977, 0.11369722324344822, 0.4860704109280778,
                 0.09299008335958303, 0.11229999691948178, 0.0005348158154161453, 0.7711636892670307,
                 0.6573053253883241, 0.39084691369185387]

            )

        n_els = 1000
        inds_2 = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals_2 = np.random.rand(len(inds_2))
        inds_2 = inds_2.T

        # `from_data` for backend flexibility
        array_2 = SparseArray.from_data(
            (
                vals_2,
                inds_2
            ),
            shape=shape
        )

        meh = array.dot(array_2.transpose((1, 0)))
        self.assertTrue(
            np.allclose(
                meh.asarray(),
                np.dot(
                    array.asarray(),
                    array_2.asarray().T
                ),
                3
            )
        )

        n_els = 1000
        inds_3 = np.unique(np.array([np.random.choice(x, n_els) for x in shape]).T, axis=0)
        vals_3 = np.random.rand(len(inds_3))
        inds_3 = inds_3.T

        # `from_data` for backend flexibility
        array_3 = SparseArray.from_data(
            (
                vals_3,
                inds_3
            ),
            shape=shape
        )

        new2 = array_2.concatenate(array_3)
        meh = np.concatenate([array_2.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new2 = array_2.concatenate(array_3, array_2)
        meh = np.concatenate([array_2.asarray(), array_3.asarray(), array_2.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat many failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new3 = array_2.concatenate(array_2, array_3, axis=1)
        meh = np.concatenate([array_2.asarray(), array_2.asarray(), array_3.asarray()], axis=1)
        self.assertEquals(new3.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new3.asarray(),
                meh
            ),
            msg="concat along 1 failed: (ref) {} vs {}".format(
                meh,
                new3.asarray()
            )
        )

        new_shape = [1, shape[1]]

        n_els = 1000
        inds_3 = np.unique(np.array([np.random.choice(x, n_els) for x in new_shape]).T, axis=0)
        vals_3 = np.random.rand(len(inds_3))
        inds_3 = inds_3.T

        # `from_data` for backend flexibility
        array_3 = SparseArray.from_data(
            (
                vals_3,
                inds_3
            ),
            shape=new_shape
        )

        new2 = array_3.concatenate(array_2)
        meh = np.concatenate([array_3.asarray(), array_2.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new2 = array_2.concatenate(array_3)
        meh = np.concatenate([array_2.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new2 = array_2.concatenate(array_3, array_2)
        meh = np.concatenate([array_2.asarray(), array_3.asarray(), array_2.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat many failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        # new3 = array_2.concatenate(array_2, array_3, axis=1)
        # meh = np.concatenate([array_2.asarray(), array_2.asarray(), array_3.asarray()], axis=1)
        # self.assertEquals(new3.shape, meh.shape)
        # self.assertTrue(
        #     np.allclose(
        #         new3.asarray(),
        #         meh
        #     ),
        #     msg="concat along 1 failed: (ref) {} vs {}".format(
        #         meh,
        #         new3.asarray()
        #     )
        # )

        array_3 = array_3[:, :2500].reshape((1, 2500))

        array_3 = array_3.reshape((
            array_3.shape[1] // 2,
            2
        ))

        new2 = array_3.concatenate(array_3)
        meh = np.concatenate([array_3.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new2 = array_3.concatenate(array_3, axis=1)
        meh = np.concatenate([array_3.asarray(), array_3.asarray()], axis=1)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        new_shape = [shape[1]]

        n_els = 1000
        inds_3 = np.unique(np.array([np.random.choice(x, n_els) for x in new_shape]).T, axis=0)
        vals_3 = np.random.rand(len(inds_3))
        inds_3 = inds_3.T

        array_3 = SparseArray.from_data(
            (
                vals_3,
                inds_3
            ),
            shape=new_shape
        )

        new2 = array_3.concatenate(array_3)
        meh = np.concatenate([array_3.asarray(), array_3.asarray()], axis=0)
        self.assertEquals(new2.shape, meh.shape)
        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )

        wtf_array1 = SparseArray.from_data(
            (
                [-0.00045906, -0.00045906, -0.00045906, -0.00045906, -0.00045906,
                 -0.00045906],
                (
                    (0, 24, 51, 78, 109, 140),
                )
            ),
            shape=(155,)

        )

        wtf_array2 = SparseArray.from_data(
            (
                [-0.00045906, -0.00045906, -0.00045906, -0.00045906],
                ([16, 53, 88, 123],)
            ),
            shape=(155,)
        )

        new2 = wtf_array1.concatenate(wtf_array2)
        meh = np.concatenate([
            wtf_array1.asarray(),
            wtf_array2.asarray()
        ])

        self.assertTrue(
            np.allclose(
                new2.asarray(),
                meh
            ),
            msg="concat failed: (ref) {} vs {}".format(
                meh,
                new2.asarray()
            )
        )
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/Sparse/SparseArray.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/Sparse/SparseArray.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/Sparse/SparseArray.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/Sparse/SparseArray.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/Sparse.py?message=Update%20Docs)