# <a id="McUtils.Numputils.TransformationMatrices.rotation_matrix">rotation_matrix</a>

```python
rotation_matrix(axis, theta): 
```

- `axis`: `Any`
    >No description...
- `theta`: `Any`
    >angle to rotate by (or Euler angles)
- `:returns`: `_`
    >No description... 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

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

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/TransformationMatrices/rotation_matrix.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/TransformationMatrices/rotation_matrix.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/TransformationMatrices/rotation_matrix.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/TransformationMatrices/rotation_matrix.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/TransformationMatrices.py?message=Update%20Docs)