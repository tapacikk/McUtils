# <a id="McUtils.Numputils.AnalyticDerivs.vec_norm_derivs">vec_norm_derivs</a>

```python
vec_norm_derivs(a, order=1, zero_thresh=None): 
```
Derivative of the norm of `a` with respect to its components
- `a`: `np.ndarray`
    >vector
- `order`: `int`
    >number of derivatives to return
- `zero_thresh`: `Any`
    >No description...
- `:returns`: `list`
    >derivative tensors 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [NormDerivs](#NormDerivs)

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

#### <a name="NormDerivs">NormDerivs</a>
```python
    def test_NormDerivs(self):
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        a = coords[(4, 5),] - coords[(5, 4),]
        na, na_da, na_daa = vec_norm_derivs(a, order=2)
        na_2 = vec_norms(a)

        self.assertEquals(tuple(na_2), tuple(na))


        norm_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_norms(vecs),
            function_shape=((None, 2, 3), (None,)),
            mesh_spacing=1.0e-4
        )

        fd_nada, fd_nadaa = norm_fd(a).derivative_tensor([1, 2])

        fd_nada = np.array([fd_nada[:3, 0], fd_nada[3:, 1]])
        fd_nadaa = np.array([fd_nadaa[:3, :3, 0], fd_nadaa[3:, 3:, 1]])

        self.assertTrue(np.allclose(na_da.flatten(), fd_nada.flatten()), msg="norm d1: {} and {} differ".format(
            na_da.flatten(), fd_nada.flatten()
        ))

        self.assertTrue(np.allclose(na_daa.flatten(), fd_nadaa.flatten(), atol=1.0e-4), msg="norm d1: {} and {} differ".format(
            na_daa.flatten(), fd_nadaa.flatten()
        ))
```

 </div>
</div>
___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/AnalyticDerivs/vec_norm_derivs.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/AnalyticDerivs/vec_norm_derivs.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/AnalyticDerivs/vec_norm_derivs.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/AnalyticDerivs/vec_norm_derivs.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/AnalyticDerivs.py?message=Update%20Docs)