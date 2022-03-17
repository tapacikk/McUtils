## <a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion">FunctionExpansion</a>
A class for handling expansions of an internal coordinate potential up to 4th order
Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian

### Properties and Methods
```python
CoordinateTransforms: type
FunctionDerivatives: type
```
<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, derivatives, transforms=None, center=None, ref=0, weight_coefficients=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FunctionExpansions.py#L24)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FunctionExpansions.py#L24?message=Update%20Docs)]
</div>


- `derivatives`: `Iterable[np.ndarray | Tensor]`
    >Derivatives of the function being expanded
- `transforms`: `Iterable[np.ndarray | Tensor] | None`
    >Jacobian and higher order derivatives in the coordinates
- `center`: `np.ndarray | None`
    >the reference point for expanding aobut
- `ref`: `float | np.ndarray`
    >the reference point value for shifting the expansion
- `weight_coefficients`: `bool`
    >whether the derivative terms need to be weighted or not

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand_function" class="docs-object-method">&nbsp;</a> 
```python
expand_function(f, point, order=4, basis=None, function_shape=None, transforms=None, weight_coefficients=True, **fd_options): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FunctionExpansions.py#L56)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FunctionExpansions.py#L56?message=Update%20Docs)]
</div>

Expands a function about a point up to the given order
- `f`: `function`
    >No description...
- `point`: `np.ndarray | CoordinateSet`
    >No description...
- `order`: `int`
    >No description...
- `basis`: `None | CoordinateSystem`
    >No description...
- `fd_options`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expansion_tensors" class="docs-object-method">&nbsp;</a> 
```python
@property
expansion_tensors(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FunctionExpansions.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FunctionExpansions.py#L?message=Update%20Docs)]
</div>

Provides the tensors that will contracted
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_expansions" class="docs-object-method">&nbsp;</a> 
```python
get_expansions(self, coords, squeeze=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FunctionExpansions.py#L109)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FunctionExpansions.py#L109?message=Update%20Docs)]
</div>


- `coords`: `np.ndarray | CoordinateSet`
    >Coordinates to evaluate the expansion at
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand" class="docs-object-method">&nbsp;</a> 
```python
expand(self, coords, squeeze=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FunctionExpansions.py#L145)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FunctionExpansions.py#L145?message=Update%20Docs)]
</div>

Returns a numerical value for the expanded coordinates
- `coords`: `np.ndarray`
    >No description...
- `:returns`: `float | np.ndarray`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, coords, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/FunctionExpansions.py#L157)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/FunctionExpansions.py#L157?message=Update%20Docs)]
</div>

## <a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion">FunctionExpansion</a>
A class for handling expansions of an internal coordinate potential up to 4th order
Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian

### Properties and Methods
```python
expand_function: method
CoordinateTransforms: type
FunctionDerivatives: type
```
<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, derivatives, transforms=None, center=None, ref=0, weight_coefficients=True): 
```

- `derivatives`: `Iterable[np.ndarray | Tensor]`
    >Derivatives of the function being expanded
- `transforms`: `Iterable[np.ndarray | Tensor] | None`
    >Jacobian and higher order derivatives in the coordinates
- `center`: `np.ndarray | None`
    >the reference point for expanding aobut
- `ref`: `float | np.ndarray`
    >the reference point value for shifting the expansion
- `weight_coefficients`: `bool`
    >whether the derivative terms need to be weighted or not

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.tensors" class="docs-object-method">&nbsp;</a>
```python
@property
tensors(self): 
```

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_expansions" class="docs-object-method">&nbsp;</a>
```python
get_expansions(self, coords, squeeze=True): 
```

- `coords`: `np.ndarray | CoordinateSet`
    >Coordinates to evaluate the expansion at
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand" class="docs-object-method">&nbsp;</a>
```python
expand(self, coords, squeeze=True): 
```
Returns a numerical value for the expanded coordinates
- `coords`: `np.ndarray`
    >No description...
- `:returns`: `float | np.ndarray`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, coords, **kw): 
```

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_tensor" class="docs-object-method">&nbsp;</a>
```python
get_tensor(self, i): 
```
Defines the overall tensors of derivatives
- `i`: `order of derivative tensor to provide`
    >No description...
- `:returns`: `Tensor`
    >No description...

### Examples


___

[Edit Examples](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Examples](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Template](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Template](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions.py?message=Update%20Docs)


<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [ExpandFunction](#ExpandFunction)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
#### <a class="collapse-link" data-toggle="collapse" href="#test-setup">Setup</a> <a class="float-right" data-toggle="collapse" href="#test-setup"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="test-setup" markdown="1">

Before we can run our examples we should get a bit of setup out of the way.
Since these examples were harvested from the unit tests not all pieces
will be necessary for all situations.
```python
try:
    from Peeves.TestUtils import *
    from Peeves import BlockProfiler
except:
    pass
from unittest import TestCase
from McUtils.Zachary import *
from McUtils.Zachary.Taylor.ZachLib import *
from McUtils.Plots import *
from McUtils.Data import *
import McUtils.Numputils as nput
from McUtils.Parallelizers import *
from McUtils.Scaffolding import Logger
import sys, h5py, math, numpy as np, itertools
```

All tests are wrapped in a test class
```python
class ZacharyTests(TestCase):
    def setUp(self):
        self.save_data = TestManager.data_gen_tests
    def __getstate__(self):
        return {}
    def __setstate__(self, state):
        pass
    def get_error(self, ref_vals, vals):
        err = np.abs(vals - ref_vals)
        cum_err = np.linalg.norm(err.flatten())
        max_err = np.max(err.flatten())
        mean_err = np.average(err.flatten())
        return err, cum_err, max_err, mean_err
    def plot_err(self, grid, ref_vals, vals, errs):
        if grid.ndim == 1:
            Plot(grid, ref_vals, figure=Plot(grid, vals))
            Plot(grid, errs[0]).show()
        elif grid.ndim == 3:
            g = (grid[:, :, 0], grid[:, :, 1])
            gg = GraphicsGrid(nrows=2, ncols=2)
            gg[0, 0] = ContourPlot(*g, ref_vals, figure=gg[0, 0])
            gg[1, 0] = ContourPlot(*g, vals, figure=gg[1, 0])
            gg[1, 1] = ContourPlot(*g, errs[0], figure=gg[1, 1])
            gg[0, 0].tight_layout()
            gg.show()
    def print_error(self, n, order, errs):
        print(
            "Order: {}.{}\nCumulative Error: {}\nMax Error: {}\nMean Error: {}".format(n, order, *errs[1:])
        )
    class harmonically_coupled_morse:
        # mass_weights = masses[:2] / np.sum(masses[:2])
        def __init__(self,
                     De_1, a_1, re_1,
                     De_2, a_2, re_2,
                     kb, b_e
                     ):
            self.De_1 = De_1
            self.a_1 = a_1
            self.re_1 = re_1
            self.De_2 = De_2
            self.a_2 = a_2
            self.re_2 = re_2
            self.kb = kb
            self.b_e = b_e

        def __call__(self, carts):
            v1 = carts[..., 1, :] - carts[..., 0, :]
            v2 = carts[..., 2, :] - carts[..., 0, :]
            r1 = nput.vec_norms(v1) - self.re_1
            r2 = nput.vec_norms(v2) - self.re_2
            bend, _ = nput.vec_angles(v1, v2)
            bend = bend - self.b_e

            return (
                    self.De_1 * (1 - np.exp(-self.a_1 * r1)) ** 2
                    + self.De_2 * (1 - np.exp(-self.a_2 * r2)) ** 2
                    + self.kb * bend**2
            )
```

 </div>
</div>

#### <a name="ExpandFunction">ExpandFunction</a>
```python
    def test_ExpandFunction(self):
        dtype = np.float32

        def sin_xy(pt):
            ax = -1 if pt.ndim > 1 else 0
            return np.prod(np.sin(pt), axis=ax)

        point = np.array([.5, .5], dtype=dtype)
        exp = FunctionExpansion.expand_function(sin_xy, point, function_shape=((2,), 0), order=4, stencil=6)

        hmm = np.vstack(
            [np.linspace(-.5, .5, 100, dtype=dtype), np.zeros((100,), dtype=dtype)]
        ).T + point[np.newaxis]
        # print(hmm)
        ref = sin_xy(hmm)
        test = exp(hmm)

        plot_error = False
        if plot_error:
            exp2 = FunctionExpansion.expand_function(sin_xy, point, function_shape=((2,), 0), order=1, stencil=5)
            g = hmm[:, 0]
            gg = GraphicsGrid(nrows=1, ncols=2, tighten=True)
            gg[0, 0] = Plot(g, exp(hmm), figure=Plot(g, sin_xy(hmm), figure=gg[0, 0]))
            gg[0, 1] = Plot(g, exp2(hmm), figure=Plot(g, sin_xy(hmm), figure=gg[0, 1]))
            gg.show()

        plot2Derror = False
        if plot2Derror:
            # -0.008557147793556821113 0.007548466137326598213 0.0019501675856203966399
            mesh = np.meshgrid(
                np.linspace(.4, .6, 100, dtype=dtype),
                np.linspace(.4, .6, 100, dtype=dtype)
            )
            grid = np.array(mesh).T
            gg = GraphicsGrid(nrows=1, ncols=2, tighten=True)
            ref2 = sin_xy(grid)
            test2 = exp(grid)
            err2 = ref2 - test2
            # print(np.min(err2), np.max(err2), np.average(np.abs(err2)))
            gg[0, 1] = ContourPlot(*mesh, ref2 - test2, figure=gg[0, 1], plot_style=dict(vmin=-.2, vmax=.2))
            gg[0, 0] = ContourPlot(*mesh, test2, figure=gg[0, 0])
            gg.show()

        self.assertEquals(exp(point), exp.ref)
        self.assertLess(np.linalg.norm(test - ref), .01)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Taylor/FunctionExpansions.py?message=Update%20Docs)