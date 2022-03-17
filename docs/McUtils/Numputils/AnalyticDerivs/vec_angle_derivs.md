# <a id="McUtils.Numputils.AnalyticDerivs.vec_angle_derivs">vec_angle_derivs</a>

```python
vec_angle_derivs(a, b, order=1, up_vectors=None, zero_thresh=None): 
```
Returns the derivatives of the angle between `a` and `b` with respect to their components
- `a`: `np.ndarray`
    >vector
- `b`: `np.ndarray`
    >vector
- `order`: `int`
    >order of derivatives to go up to
- `zero_thresh`: `float | None`
    >threshold for what is zero in a vector norm
- `:returns`: `list`
    >derivative tensors 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [AngleDerivs](#AngleDerivs)
- [AngleDerivScan](#AngleDerivScan)

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

#### <a name="AngleDerivs">AngleDerivs</a>
```python
    def test_AngleDerivs(self):
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        a = coords[4] - coords[5]
        b = coords[6] - coords[5]
        ang, dang, ddang = vec_angle_derivs(np.array([a, b]),
                                            np.array([b, a]), order=2)
        ang_2 = vec_angles(a, b)[0]

        self.assertEquals(ang_2, ang.flatten()[0])

        ang_fd = FiniteDifferenceDerivative(
            lambda vecs: vec_angles(vecs[..., 0, :], vecs[..., 1, :])[0],
            function_shape=((None, 2, 3), 0),
            mesh_spacing=1.0e-4
        )

        fd_ang_1, fd_dang_1 = ang_fd([a, b]).derivative_tensor([1, 2])
        fd_ang_2, fd_dang_2 = ang_fd([b, a]).derivative_tensor([1, 2])

        fd_ang = np.array([fd_ang_1, fd_ang_2])
        fd_dang = np.array([fd_dang_1, fd_dang_2])

        self.assertTrue(np.allclose(dang.flatten(), fd_ang.flatten()), msg="ang d1: {} and {} differ".format(
            fd_ang.flatten(), fd_ang.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([ddang[:, 0, 0], ddang[:, 0, 1]], axis=2),
                np.concatenate([ddang[:, 1, 0], ddang[:, 1, 1]], axis=2)
            ],
            axis=1
        )

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(d2_flat[0], fd_dang[0])))
        self.assertTrue(np.allclose(d2_flat.flatten(), fd_dang.flatten(), atol=1.0e-2), msg="ang d2: {} and {} differ ({})".format(
            d2_flat.flatten(), fd_dang.flatten(), d2_flat.flatten() - fd_dang.flatten()
        ))
```
#### <a name="AngleDerivScan">AngleDerivScan</a>
```python
    def test_AngleDerivScan(self):
        np.random.seed(0)
        # a = np.random.rand(3) * 2 # make it longer to avoid stability issues
        a = np.array([1, 0, 0])

        fd = FiniteDifferenceDerivative(
                lambda vecs: vec_angle_derivs(vecs[..., 0, :], vecs[..., 1, :], up_vectors=up)[1],
                function_shape=((None, 2, 3), 0),
                mesh_spacing=1.0e-4
            )

        data = {"rotations":[], 'real_angles':[], "angles":[], 'derivs':[], 'derivs2':[], 'derivs_num2':[]}
        for q in np.linspace(-np.pi, np.pi, 601):
            up = np.array([0, 0, 1])
            r = rotation_matrix(up, q)
            b = np.dot(r, a)
            ang, deriv, deriv_2 = vec_angle_derivs(a, b, up_vectors=up, order=2)
            data['rotations'].append(q)
            data['real_angles'].append(vec_angles(a, b, up_vectors=up)[0])
            data['angles'].append(ang.tolist())
            data['derivs'].append(deriv.tolist())
            data['derivs2'].append(deriv_2.tolist())

            data['derivs_num2'].append(fd(np.array([a, b])).derivative_tensor(1).tolist())
```

 </div>
</div>
___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/AnalyticDerivs/vec_angle_derivs.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/AnalyticDerivs/vec_angle_derivs.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/AnalyticDerivs/vec_angle_derivs.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/AnalyticDerivs/vec_angle_derivs.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/AnalyticDerivs.py?message=Update%20Docs)