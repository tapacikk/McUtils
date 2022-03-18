# <a id="McUtils.Zachary.Interpolator">McUtils.Zachary.Interpolator</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/tree/master/Zachary/Interpolator)]
</div>
    
Sets up a general Interpolator class that looks like Mathematica's InterpolatingFunction class

<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[Interpolator](Interpolator/Interpolator.md)   
</div>
   <div class="col" markdown="1">
[Extrapolator](Interpolator/Extrapolator.md)   
</div>
   <div class="col" markdown="1">
[ProductGridInterpolator](Interpolator/ProductGridInterpolator.md)   
</div>
</div>
</div>

## Examples



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [Interpolator1D](#Interpolator1D)
- [InterpolatorExtrapolator2D](#InterpolatorExtrapolator2D)
- [InterpolatorExtrapolator1D](#InterpolatorExtrapolator1D)
- [RegularInterpolator3D](#RegularInterpolator3D)
- [IrregularInterpolator3D](#IrregularInterpolator3D)

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

#### <a name="Interpolator1D">Interpolator1D</a>
```python
    def test_Interpolator1D(self):

        def test_fn(grid):
            return np.sin(grid)
        sin_grid = np.linspace(0, 1, 12)
        grid = sin_grid

        sin_vals = test_fn(sin_grid)
        interp = Interpolator(grid, sin_vals)

        test_points = np.random.uniform(0, 1, size=(100,))
        test_vals = test_fn(test_points)
        interp_vals = interp(test_points)
        self.assertTrue(np.allclose(interp_vals, test_vals, atol=5e-5))
```
#### <a name="InterpolatorExtrapolator2D">InterpolatorExtrapolator2D</a>
```python
    def test_InterpolatorExtrapolator2D(self):

        def test_fn(grid):
            return (
                .25 * (np.sin(grid[..., 0])**2) * np.cos(grid[..., 1])
                + .5 * np.sin(grid[..., 0])*np.cos(grid[..., 1])
                + .25 * np.sin(grid[..., 0]) * (np.cos(grid[..., 1])**2)
            )

        cos_grid = np.linspace(0, np.pi, 8)
        sin_grid = np.linspace(0, np.pi, 11)
        grids = [sin_grid, cos_grid]

        grid = np.moveaxis(np.array(np.meshgrid(*grids, indexing='ij')), 0, 2)
        vals = test_fn(grid)

        interp = Interpolator(grid, vals)
        lin_ext_interp = Interpolator(grid, vals, extrapolator=Interpolator.DefaultExtrapolator, extrapolation_order=1)

        g = GraphicsGrid(nrows=1, ncols=3,
                         figure_label='Extrapolation tests',
                         spacings=(70, 0)
                         )
        g.padding_top = 40
        g[0, 0].plot_label = "Analytic"
        g[0, 1].plot_label = "Cubic Extrapolation"
        g[0, 2].plot_label = "Linear Extrapolation"

        big_grid = np.array(np.meshgrid(
            np.linspace(-np.pi/2, 3*np.pi/2, 500),
            np.linspace(-np.pi/2, 3*np.pi/2, 500)
        ))
        eval_grid = np.moveaxis(big_grid, 0, 2)#.transpose((1, 0, 2))

        inner_grid = np.array(np.meshgrid(
            np.linspace(0, np.pi, 500),
            np.linspace(0, np.pi, 500)
        ))
        inner_eval_grid = np.moveaxis(inner_grid, 0, 2)

        surf = test_fn(eval_grid)
        ContourPlot(*big_grid, surf, figure=g[0, 0],
                    plot_style=dict(vmin=np.min(surf), vmax=np.max(surf))
                    )
        ContourPlot(*big_grid, interp(eval_grid), figure=g[0, 1],
                    plot_style=dict(vmin=np.min(surf), vmax=np.max(surf)))
        ContourPlot(*inner_grid, interp(inner_eval_grid), figure=g[0, 1],
                    plot_style=dict(vmin=np.min(surf), vmax=np.max(surf)))
        ContourPlot(*big_grid, lin_ext_interp(eval_grid), figure=g[0, 2],
                    plot_style=dict(vmin=np.min(surf), vmax=np.max(surf)))
        ContourPlot(*inner_grid, lin_ext_interp(inner_eval_grid), figure=g[0, 2],
                    plot_style=dict(vmin=np.min(surf), vmax=np.max(surf)))
        ScatterPlot(*np.moveaxis(grid, 2, 0), figure=g[0, 0],
                    plot_style=dict(color='red')
                    )
        ScatterPlot(*np.moveaxis(grid, 2, 0), figure=g[0, 1],
                    plot_style=dict(color='red')
                    )
        ScatterPlot(*np.moveaxis(grid, 2, 0), figure=g[0, 2],
                    plot_style=dict(color='red')
                    )

        g.show()
```
#### <a name="InterpolatorExtrapolator1D">InterpolatorExtrapolator1D</a>
```python
    def test_InterpolatorExtrapolator1D(self):

        sin_grid = np.linspace(0, np.pi, 5)
        sin_vals = np.sin(sin_grid)
        interp = Interpolator(sin_grid, sin_vals,
                              extrapolator=Interpolator.DefaultExtrapolator, extrapolation_order=1)
        default_interp = Interpolator(sin_grid, sin_vals)

        test_points = np.random.uniform(-np.pi, 2*np.pi, size=(20,))
        test_vals = np.sin(test_points)
        interp_vals = interp(test_points)
```
#### <a name="RegularInterpolator3D">RegularInterpolator3D</a>
```python
    def test_RegularInterpolator3D(self):

        def test_fn(grid):
            return np.sin(grid[..., 0])*np.cos(grid[..., 1]) - np.sin(grid[..., 2])

        cos_grid = np.linspace(0, 1, 8)
        sin_grid1 = np.linspace(0, 1, 11)
        sin_grid = np.linspace(0, 1, 12)
        grids = [sin_grid, sin_grid1, cos_grid]

        grid = np.moveaxis(np.array(np.meshgrid(*grids, indexing='ij')), 0, 3)
        vals = test_fn(grid)

        interp = Interpolator(grid, vals)

        test_points = np.random.uniform(0, 1, size=(100000, 3))
        test_vals = test_fn(test_points)
        interp_vals = interp(test_points)
        self.assertTrue(np.allclose(interp_vals, test_vals, atol=5e-5))
```
#### <a name="IrregularInterpolator3D">IrregularInterpolator3D</a>
```python
    def test_IrregularInterpolator3D(self):

        def test_fn(grid):
            return np.sin(grid[..., 0])*np.cos(grid[..., 1]) - np.sin(grid[..., 2])

        np.random.seed(3)
        grid = np.random.uniform(0, 1, size=(10000, 3))
        vals = test_fn(grid)
        interp = Interpolator(grid, vals)

        test_points = np.random.uniform(0, 1, size=(1000, 3))
        test_vals = test_fn(test_points)
        interp_vals = interp(test_points)

        self.assertTrue(np.allclose(interp_vals, test_vals, atol=5e-3), msg='max diff: {}'.format(
            np.max(np.abs(interp_vals - test_vals))
        ))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/Interpolator.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/Interpolator.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/Interpolator.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/Interpolator.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator/__init__.py?message=Update%20Docs)