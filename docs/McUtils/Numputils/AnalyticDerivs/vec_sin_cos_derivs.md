# <a id="McUtils.Numputils.AnalyticDerivs.vec_sin_cos_derivs">vec_sin_cos_derivs</a>

```python
vec_sin_cos_derivs(a, b, order=1, check_derivatives=False, zero_thresh=None): 
```
Derivative of `sin(a, b)` and `cos(a, b)` with respect to both vector components
- `a`: `np.ndarray`
    >vector
- `order`: `int`
    >number of derivatives to return
- `zero_thresh`: `None | float`
    >threshold for when a norm should be called 0. for numerical reasons
- `:returns`: `list`
    >derivative tensors 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [SinCosDerivs](#SinCosDerivs)

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

#### <a name="SinCosDerivs">SinCosDerivs</a>
```python
    def test_SinCosDerivs(self):
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        # sin_derivs_extra, cos_derivs_extra = vec_sin_cos_derivs( # just here to make sure the shape works
        #     coords[(1, 2, 3, 4),] - coords[(0, 1, 2, 3),],
        #     coords[(2, 3, 4, 5),] - coords[(0, 1, 2, 3),],
        #     order=2)

        a = coords[6] - coords[5]
        b = coords[4] - coords[5]

        sin_derivs, cos_derivs = vec_sin_cos_derivs(np.array([a, b]), np.array([b, a]), order=2)

        cos_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_cos(vecs[..., 0, :], vecs[..., 1, :]),
            function_shape=((None, 2, 3), 0),
            mesh_spacing=1.0e-7
        )
        cosDeriv_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_sin_cos_derivs(vecs[..., 0, :], vecs[..., 1, :], order=1)[1][1].squeeze(),
            function_shape=((None, 2, 3), (None, 2, 3)),
            mesh_spacing=1.0e-7
        )
        cos_fd22_1, = cosDeriv_fd(np.array([a, b])).derivative_tensor([1])
        cos_fd22_2, = cosDeriv_fd(np.array([b, a])).derivative_tensor([1])
        cos_fd22 = np.array([cos_fd22_1, cos_fd22_2])

        cos_fd1_1, cos_fd2_1 = cos_fd(np.array([a, b])).derivative_tensor([1, 2])
        cos_fd1_2, cos_fd2_2 = cos_fd(np.array([b, a])).derivative_tensor([1, 2])
        cos_fd1 = np.array([cos_fd1_1, cos_fd1_2])
        cos_fd2 = np.array([cos_fd2_1, cos_fd2_2])

        sin_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_sins(vecs[..., 0, :], vecs[..., 1, :]),
            function_shape=((None, 2, 3), 0),
            mesh_spacing=1.0e-7
        )
        sinDeriv_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_sin_cos_derivs(vecs[..., 0, :], vecs[..., 1, :], order=1)[0][1].squeeze(),
            function_shape=((None, 2, 3), (None, 2, 3)),
            mesh_spacing=1.0e-7
        )
        sin_fd22_1, = sinDeriv_fd(np.array([a, b])).derivative_tensor([1])
        sin_fd22_2, = sinDeriv_fd(np.array([b, a])).derivative_tensor([1])
        sin_fd22 = np.array([sin_fd22_1, sin_fd22_2])
        sin_fd1_1, sin_fd2_1 = sin_fd(np.array([a, b])).derivative_tensor([1, 2])
        sin_fd1_2, sin_fd2_2 = sin_fd(np.array([b, a])).derivative_tensor([1, 2])
        sin_fd1 = np.array([sin_fd1_1, sin_fd1_2])
        sin_fd2 = np.array([sin_fd2_1, sin_fd2_2])

        s, s1, s2 = sin_derivs
        c, c1, c2 = cos_derivs

        # raise Exception(sin_fd1, s1)

        self.assertTrue(np.allclose(s1.flatten(), sin_fd1.flatten()), msg="sin d1: {} and {} differ".format(
            s1.flatten(), sin_fd1.flatten()
        ))

        self.assertTrue(np.allclose(c1.flatten(), cos_fd1.flatten()), msg="cos d1: {} and {} differ".format(
            c1.flatten(), cos_fd1.flatten()
        ))

        # raise Exception("\n", c2[0, 0], "\n", cos_fd2[:3, :3])
        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(c2[0, 0], cos_fd2[:3, :3])))
        c2_flat = np.concatenate(
            [
                np.concatenate([c2[:, 0, 0], c2[:, 0, 1]], axis=2),
                np.concatenate([c2[:, 1, 0], c2[:, 1, 1]], axis=2)
            ],
            axis=1
        )

        s2_flat = np.concatenate(
            [
                np.concatenate([s2[:, 0, 0], s2[:, 0, 1]], axis=2),
                np.concatenate([s2[:, 1, 0], s2[:, 1, 1]], axis=2)
            ],
            axis=1
        )

        self.assertTrue(np.allclose(s2_flat.flatten(), sin_fd22.flatten()), msg="sin d2: {} and {} differ".format(
            s2_flat.flatten(), sin_fd22.flatten()
        ))
        self.assertTrue(np.allclose(c2_flat.flatten(), cos_fd22.flatten()), msg="cos d2: {} and {} differ".format(
            c2_flat.flatten(), cos_fd22.flatten()
        ))
```

 </div>
</div>
___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/AnalyticDerivs/vec_sin_cos_derivs.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/AnalyticDerivs/vec_sin_cos_derivs.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/AnalyticDerivs/vec_sin_cos_derivs.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/AnalyticDerivs/vec_sin_cos_derivs.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/AnalyticDerivs.py?message=Update%20Docs)