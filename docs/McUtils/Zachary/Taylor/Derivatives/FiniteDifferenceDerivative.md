## <a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative">FiniteDifferenceDerivative</a>
Provides derivatives for a function (scalar or vector valued).
Can be indexed into or the entire tensor of derivatives may be requested.
The potential for optimization undoubtedly exists, but the idea is to provide as _simple_ an interface as possible.
Robustification needs to be done, but is currently used in `CoordinateSystem.jacobian` to good effect.

### Properties and Methods
<a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, f, function_shape=(0, 0), parallelizer=None, logger=None, **fd_opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/Derivatives.py#L21)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/Derivatives.py#L21?message=Update%20Docs)]
</div>


- `f`: `FunctionSpec | callable`
    >the function we would like to take derivatives of
- `function_shape`: `Iterable[Iterable[int] | int] | None`
    >the shape of the function we'd like to take the derivatives of
- `fd_opts`: `Any`
    >the options to pass to the finite difference function

<a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, *args, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/Derivatives.py#L45)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/Derivatives.py#L45?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative.derivatives" class="docs-object-method">&nbsp;</a> 
```python
derivatives(self, center, displacement_function=None, prep=None, lazy=None, mesh_spacing=None, **fd_opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Taylor/Derivatives.py#L48)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Taylor/Derivatives.py#L48?message=Update%20Docs)]
</div>

Generates a differencer object that can be used to get derivs however your little heart desires
- `center`: `np.ndarray`
    >the center point around which to generate differences
- `displacement_function`: `Any`
    >No description...
- `mesh_spacing`: `Any`
    >No description...
- `prep`: `Any`
    >No description...
- `fd_opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

## <a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative">FiniteDifferenceDerivative</a>
Provides derivatives for a function (scalar or vector valued).
Can be indexed into or the entire tensor of derivatives may be requested.
The potential for optimization undoubtedly exists, but the idea is to provide as _simple_ an interface as possible.
Robustification needs to be done, but is currently used in `CoordinateSystem.jacobian` to good effect.

### Properties and Methods
<a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, f, function_shape=(0, 0), **fd_opts): 
```

- `f`: `FunctionSpec | callable`
    >the function we would like to take derivatives of
- `function_shape`: `Iterable[Iterable[int] | int] | None`
    >the shape of the function we'd like to take the derivatives of
- `fd_opts`: `Any`
    >the options to pass to the finite difference function

<a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, *args, **opts): 
```

<a id="McUtils.Zachary.Taylor.Derivatives.FiniteDifferenceDerivative.derivatives" class="docs-object-method">&nbsp;</a>
```python
derivatives(self, center, displacement_function=None, prep=None, lazy=None, mesh_spacing=None, **fd_opts): 
```
Generates a differencer object that can be used to get derivs however your little heart desires
- `center`: `np.ndarray`
    >the center point around which to generate differences
- `displacement_function`: `Any`
    >No description...
- `mesh_spacing`: `Any`
    >No description...
- `prep`: `Any`
    >No description...
- `fd_opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples


___

[Edit Examples](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/examples/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) or 
[Create New Examples](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/examples/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) <br/>
[Edit Template](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/templates/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) or 
[Create New Template](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/templates/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/Derivatives.py?message=Update%20Docs)


<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [FDDeriv](#FDDeriv)
- [FDDeriv2](#FDDeriv2)
- [FiniteDifferenceParallelism](#FiniteDifferenceParallelism)

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

#### <a name="FDDeriv">FDDeriv</a>
```python
    def test_FDDeriv(self):
        # ggrid = GraphicsGrid(nrows=1, ncols=2)

        fdf = FiniteDifferenceDerivative(np.sin, stencil = 5)
        grid = np.linspace(0, 2*np.pi, 5)
        diff_fun = fdf(grid, mesh_spacing = .1)
        d = np.array([-.2, -.1, 0, .1, .2])
        raw = np.array([
            finite_difference(
                c + d,
                np.sin(c + d),
                (1,),
                stencil=5, only_center = True, end_point_accuracy=0
                ) for c in grid
        ])

        self.assertLess(np.linalg.norm(diff_fun[0] - raw), .00001)
        self.assertIsInstance(diff_fun[0, 0, 0][0], (float,))
```
#### <a name="FDDeriv2">FDDeriv2</a>
```python
    def test_FDDeriv2(self):
        def test(gps):
            x = gps[..., 0]
            y = gps[..., 1]

            return x * np.sin(y)

        fdf = FiniteDifferenceDerivative(test, function_shape=((2, ), 1), stencil = 5)

        npts = 25
        grid = np.linspace(0, 2*np.pi, npts)
        grid2D = np.meshgrid(grid, grid)
        gps = np.array(grid2D).T
        diff_fun = fdf(gps)

        diff_fun[0, 0]

        self.assertEquals(diff_fun[0, 0].shape, (npts, npts))

        self.assertEquals(diff_fun[0, :].shape, (1, 2, npts, npts)) # should I implement this with a squeeze...?
        self.assertEquals(np.linalg.norm((diff_fun[:, :][1, 0] - diff_fun[1, 0]).flatten()), 0.)
```
#### <a name="FiniteDifferenceParallelism">FiniteDifferenceParallelism</a>
```python
    def test_FiniteDifferenceParallelism(self):

        re_1 = 0.9575
        re_2 = 0.9575
        b_e = np.deg2rad(104.5)
        internals = [
            [0, -1, -1, -1],
            [1, 0, -1, -1],
            [2, 0, 1, -1]
        ]
        # internals = None
        coords = np.array([
                [0.000000, 0.000000, 0.000000],
                [re_1, 0.000000, 0.000000],
                np.dot(
                    nput.rotation_matrix([0, 0, 1], b_e),
                    [re_2, 0.000000, 0.000000]
                )
            ])

        erg2h = UnitsData.convert("Ergs", "Hartrees")
        cm2borh = UnitsData.convert("InverseAngstroms", "InverseBohrRadius")
        De_1 = 8.84e-12 * erg2h
        a_1 = 2.175 * cm2borh

        De_2 = 8.84e-12 * erg2h
        a_2 = 2.175 * cm2borh

        hz2h = UnitsData.convert("Hertz", "Hartrees")
        kb = 7.2916e14 / (2 * np.pi) * hz2h

        morse = self.harmonically_coupled_morse(
            De_1, a_1, re_1,
            De_2, a_2, re_2,
            kb, b_e
        )

        deriv_gen = FiniteDifferenceDerivative(morse,
                                               function_shape=((None, None), 0),
                                               mesh_spacing=1e-3,
                                               stencil=9,
                                               logger=Logger(),
                                               parallelizer=MultiprocessingParallelizer()#, verbose=True)
                                               ).derivatives(coords)

        with BlockProfiler("With parallelizer"):
            pot_derivs = deriv_gen.derivative_tensor([1, 2, 3, 4, 5, 6, 7])

        deriv_gen = FiniteDifferenceDerivative(morse,
                                               function_shape=((None, None), 0),
                                               mesh_spacing=1e-3,
                                               logger=Logger(),
                                               stencil=9,
                                               ).derivatives(coords)

        with BlockProfiler("Without parallelizer"):
            pot_derivs = deriv_gen.derivative_tensor([1, 2, 3, 4, 5, 6])
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Zachary/Taylor/Derivatives/FiniteDifferenceDerivative.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/Taylor/Derivatives.py?message=Update%20Docs)