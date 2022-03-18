# <a id="McUtils.Zachary.TensorDerivativeConverter">McUtils.Zachary.TensorDerivativeConverter</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/tree/master/Zachary/TensorDerivativeConverter)]
</div>
    


<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[TensorDerivativeConverter](TensorDerivativeConverter/TensorDerivativeConverter.md)   
</div>
   <div class="col" markdown="1">
[TensorExpansionTerms](TensorDerivativeConverter/TensorExpansionTerms.md)   
</div>
</div>
</div>




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [TensorConversion](#TensorConversion)

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

#### <a name="TensorConversion">TensorConversion</a>
```python
    def test_TensorConversion(self):

        n_Q = 10
        n_X = 20
        np.random.seed(1)
        x_derivs = [
            np.random.rand(n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_X),
            np.random.rand(n_Q, n_Q, n_Q, n_Q, n_X)
        ]
        v_derivs = [
            np.random.rand(n_X),
            np.random.rand(n_X, n_X),
            np.random.rand(n_X, n_X, n_X),
            np.random.rand(n_X, n_X, n_X, n_X)
        ]
        xv_derivs = [
            [None, np.random.rand(n_Q, n_X, n_X)],
            [None, np.random.rand(n_Q, n_Q, n_X, n_X)]
            ]

        t = TensorDerivativeConverter(x_derivs, v_derivs)
        new = t.convert()
        self.assertEquals(len(new), 4)
        self.assertEquals(new[1].shape, (n_Q, n_Q))

        t2 = TensorDerivativeConverter(x_derivs, v_derivs, mixed_terms=xv_derivs)
        new = t2.convert()
        self.assertEquals(len(new), 4)
        self.assertEquals(new[1].shape, (n_Q, n_Q))


        trace_derivs = TensorExpansionTerms([0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
        t1 = trace_derivs.QX(1).tr().dQ()
        self.assertEquals(str(t1), 'Tr[Q[2],2+3]')

        t1 = trace_derivs.QX(1).det().dQ().dQ()

        t1 = trace_derivs.QX(1).inverse().dQ()
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/TensorDerivativeConverter.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/TensorDerivativeConverter.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/TensorDerivativeConverter.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/TensorDerivativeConverter.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter/__init__.py?message=Update%20Docs)