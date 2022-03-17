# <a id="McUtils.Numputils.VectorOps.vec_norms">vec_norms</a>

```python
vec_norms(vecs, axis=-1): 
```

- `vecs`: `np.ndarray`
    >No description...
- `axis`: `int`
    >No description...
- `:returns`: `_`
    >No description... 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [PtsDistDeriv](#PtsDistDeriv)
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

#### <a name="PtsDistDeriv">PtsDistDeriv</a>
```python
    def test_PtsDistDeriv(self):
        # need some proper values to test this against...
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        dists, derivs, derivs_2 = dist_deriv(coords, [5, 4], [4, 5], order=2)

        dist = dists[0]; deriv = derivs[:, 0, :]; deriv_2 = derivs_2[:, :, 0, :, :]
        dists2 = vec_norms(coords[4] - coords[5])

        self.assertEquals(dists2, dist)
        # raise Exception(dist, dists2)

        fd = FiniteDifferenceDerivative(
            lambda pt: vec_norms(pt[..., 1, :] - pt[..., 0, :]),
            function_shape=((None, 3), 0),
            mesh_spacing=1.0e-5
        )

        fd1, fd2 = fd(coords[(5, 4),]).derivative_tensor([1, 2])

        self.assertTrue(np.allclose(deriv.flatten(), fd1.flatten()), msg="{} and {} aren't close".format(
            deriv.flatten(), fd1.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([deriv_2[0, 0], deriv_2[0, 1]], axis=1),
                np.concatenate([deriv_2[1, 0], deriv_2[1, 1]], axis=1)
            ],
            axis=0
        )

        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(d2_flat, fd2)))

        self.assertTrue(np.allclose(d2_flat.flatten(), fd2.flatten(), atol=1.0e-3), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), fd2.flatten()
        ))
```
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

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/VectorOps/vec_norms.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/VectorOps/vec_norms.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/VectorOps/vec_norms.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/VectorOps/vec_norms.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/VectorOps.py?message=Update%20Docs)