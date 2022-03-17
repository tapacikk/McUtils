# <a id="McUtils.Numputils.AnalyticDerivs.angle_deriv">angle_deriv</a>

```python
angle_deriv(coords, i, j, k, order=1, zero_thresh=None): 
```
Gives the derivative of the angle between i, j, and k with respect to the Cartesians
- `coords`: `np.ndarray`
    >No description...
- `i`: `int | Iterable[int]`
    >index of the central atom
- `j`: `int | Iterable[int]`
    >index of one of the outside atoms
- `k`: `int | Iterable[int]`
    >index of the other outside atom
- `:returns`: `np.ndarray`
    >derivatives of the angle with respect to atoms i, j, and k 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [PtsAngleDeriv](#PtsAngleDeriv)

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

#### <a name="PtsAngleDeriv">PtsAngleDeriv</a>
```python
    def test_PtsAngleDeriv(self):
        # need some proper values to test this against...
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        angs, derivs, derivs_2 = angle_deriv(coords, [5, 5], [4, 6], [6, 4], order=2)

        ang = angs[0]; deriv = derivs[:, 0, :]; deriv_2 = derivs_2[:, :, 0, :, :]
        ang2 = vec_angles(coords[4] - coords[5], coords[6] - coords[5])[0]

        self.assertEquals(ang2, ang)

        fd = FiniteDifferenceDerivative(
            lambda pt: vec_angles(pt[..., 1, :] - pt[..., 0, :], pt[..., 2, :] - pt[..., 0, :])[0],
            function_shape=((None, 3), 0),
            mesh_spacing=1.0e-5
        )

        fd1, fd2 = fd(coords[(5, 4, 6),]).derivative_tensor([1, 2])

        self.assertTrue(np.allclose(deriv.flatten(), fd1.flatten()), msg="{} and {} aren't close".format(
            deriv.flatten(), fd1.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([deriv_2[0, 0], deriv_2[0, 1], deriv_2[0, 2]], axis=1),
                np.concatenate([deriv_2[1, 0], deriv_2[1, 1], deriv_2[1, 2]], axis=1),
                np.concatenate([deriv_2[2, 0], deriv_2[2, 1], deriv_2[2, 2]], axis=1)
            ],
            axis=0
        )

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(d2_flat, fd2)))

        self.assertTrue(np.allclose(d2_flat.flatten(), fd2.flatten(), atol=1.0e-3), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), fd2.flatten()
        ))
```

 </div>
</div>
___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/AnalyticDerivs/angle_deriv.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/AnalyticDerivs.py?message=Update%20Docs)