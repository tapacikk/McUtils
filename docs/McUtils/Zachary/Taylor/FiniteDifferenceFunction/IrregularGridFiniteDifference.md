## <a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference">IrregularGridFiniteDifference</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FiniteDifferenceFunction.py#L508)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py#L508?message=Update%20Docs)]
</div>

Defines a finite difference over an irregular grid

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, grid, order, stencil=None, accuracy=2, end_point_accuracy=2, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FiniteDifferenceFunction.py#L512)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py#L512?message=Update%20Docs)]
</div>


- `grid`: `np.ndarray`
    >the grid to get the weights from
- `order`: `int`
    >the order of the derivative to take
- `stencil`: `int | None`
    >the number of stencil points to add
- `accuracy`: `int | None`
    >the approximate accuracy to target with the method
- `end_point_accuracy`: `int | None`
    >the extra number of stencil points to add to the end points
- `kw`: `Any`
    >options passed through to the `FiniteDifferenceMatrix`

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference.get_grid_slices" class="docs-object-method">&nbsp;</a> 
```python
get_grid_slices(grid, stencil): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FiniteDifferenceFunction.py#L539)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py#L539?message=Update%20Docs)]
</div>


- `grid`: `Any`
    >No description...
- `stencil`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference.get_weights" class="docs-object-method">&nbsp;</a> 
```python
get_weights(m, z, x): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FiniteDifferenceFunction.py#L552)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py#L552?message=Update%20Docs)]
</div>

Extracts the grid weights for an unevenly spaced grid based off of the algorithm outlined by
        Fronberger in https://pdfs.semanticscholar.org/8bf5/912bde884f6bd4cfb4991ba3d077cace94c0.pdf
- `m`: `Any`
    >highest derivative order
- `z`: `Any`
    >center of the derivatives
- `X`: `Any`
    >grid of points

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference.finite_difference_data" class="docs-object-method">&nbsp;</a> 
```python
finite_difference_data(grid, order, stencil, end_point_precision): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FiniteDifferenceFunction.py#L569)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py#L569?message=Update%20Docs)]
</div>

Constructs a finite-difference function that computes the nth derivative with a given width
- `deriv`: `Any`
    >No description...
- `accuracy`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>

## <a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference">IrregularGridFiniteDifference</a>
Defines a finite difference over an irregular grid

### Properties and Methods
```python
finite_difference_data: method
```
<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, grid, order, stencil=None, accuracy=2, end_point_accuracy=2, **kw): 
```

- `grid`: `np.ndarray`
    >the grid to get the weights from
- `order`: `int`
    >the order of the derivative to take
- `stencil`: `int | None`
    >the number of stencil points to add
- `accuracy`: `int | None`
    >the approximate accuracy to target with the method
- `end_point_accuracy`: `int | None`
    >the extra number of stencil points to add to the end points
- `kw`: `Any`
    >options passed through to the `FiniteDifferenceMatrix`

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference.get_grid_slices" class="docs-object-method">&nbsp;</a>
```python
get_grid_slices(grid, stencil): 
```

- `grid`: `Any`
    >No description...
- `stencil`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FiniteDifferenceFunction.IrregularGridFiniteDifference.get_weights" class="docs-object-method">&nbsp;</a>
```python
get_weights(m, z, x): 
```
Extracts the grid weights for an unevenly spaced grid based off of the algorithm outlined by
        Fronberger in https://pdfs.semanticscholar.org/8bf5/912bde884f6bd4cfb4991ba3d077cace94c0.pdf
- `m`: `Any`
    >highest derivative order
- `z`: `Any`
    >center of the derivatives
- `X`: `Any`
    >grid of points

### Examples


___

[Edit Examples](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) or 
[Create New Examples](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) <br/>
[Edit Template](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/templates/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) or 
[Create New Template](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/templates/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py?message=Update%20Docs)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [uneven_weights](#uneven_weights)

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

#### <a name="uneven_weights">uneven_weights</a>
```python
    def test_uneven_weights(self):
        import numpy as np
        weights = IrregularGridFiniteDifference.get_weights
        uweights =  [
            weights(2, 0, np.array([-2, -1, 0, 1, 2])),
            weights(1, 0, np.array([-3/2, -1/2, 1/2, 3/2])),
            weights(1, 1, np.array([-3, -2, -1, 0, 1]))
            #weights(3, 1/2, np.array([0, 1/3, 1, 2, 7/2, 6]))
            ]
        eweights = RegularGridFiniteDifference.get_weights
        targ_weights = [
            eweights(2, 2,   4),
            eweights(1, 3/2, 3),
            eweights(1, 4,   4)
        ]
        passed = True
        for e, w in zip(targ_weights, uweights):
            norm = np.linalg.norm(e-w[-1])
            if norm > .000001:
                passed = False
                print((norm, e, w[-1]), file=sys.stderr)
        self.assertIs(passed, True)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Taylor/FiniteDifferenceFunction/IrregularGridFiniteDifference.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FiniteDifferenceFunction.py#L508?message=Update%20Docs)