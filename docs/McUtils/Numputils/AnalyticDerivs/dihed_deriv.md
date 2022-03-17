# <a id="McUtils.Numputils.AnalyticDerivs.dihed_deriv">dihed_deriv</a>

```python
dihed_deriv(coords, i, j, k, l, order=1, zero_thresh=None, zero_point_step_size=0.0001): 
```
Gives the derivative of the dihedral between i, j, k, and l with respect to the Cartesians
    Currently gives what are sometimes called the `psi` angles.
    Can also support more traditional `phi` angles by using a different angle ordering
- `coords`: `np.ndarray`
    >No description...
- `i`: `int | Iterable[int]`
    >No description...
- `j`: `int | Iterable[int]`
    >No description...
- `k`: `int | Iterable[int]`
    >No description...
- `l`: `int | Iterable[int]`
    >No description...
- `:returns`: `np.ndarray`
    >derivatives of the dihedral with respect to atoms i, j, k, and l 




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [ProblemPtsAllDerivs](#ProblemPtsAllDerivs)
- [PtsDihedralsDeriv](#PtsDihedralsDeriv)

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

#### <a name="ProblemPtsAllDerivs">ProblemPtsAllDerivs</a>
```python
    def test_ProblemPtsAllDerivs(self):
        import McUtils.Numputils.Options as NumOpts

        NumOpts.zero_placeholder = np.inf

        coords = self.problem_coords

        # dists, dist_derivs, dist_derivs_2 = dist_deriv(coords, [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], order=2)
        # angs, ang_derivs, ang_derivs_2 = angle_deriv(coords,
        #                                              [0, 1, 2, 3, 4, 5],
        #                                              [1, 2, 3, 4, 5, 0],
        #                                              [2, 3, 4, 5, 0, 1],
        #                                              order=2
        #                                              )
        # diheds, dihed_derivs, dihed_derivs_2 = dihed_deriv(coords,
        #                                                [0, 1, 2, 3, 4, 5],
        #                                                [1, 2, 3, 4, 5, 0],
        #                                                [2, 3, 4, 5, 0, 1],
        #                                                [3, 4, 5, 0, 1, 2],
        #                                                order=2
        #                                                )

        diheds, dihed_derivs, dihed_derivs_2 = dihed_deriv(coords,
                                                           [3],
                                                           [4],
                                                           [5],
                                                           [0],
                                                           order=2
                                                           )
```
#### <a name="PtsDihedralsDeriv">PtsDihedralsDeriv</a>
```python
    def test_PtsDihedralsDeriv(self):
        # need some proper values to test this against...
        np.random.seed(0)
        coords = np.random.rand(16, 3)

        angs, derivs, derivs_2 = dihed_deriv(coords, [4, 7], [5, 6], [6, 5], [7, 4], order=2)
        ang = angs[0]; deriv = derivs[:, 0, :]; deriv_2 = derivs_2[:, :, 0, :, :]
        ang2 = pts_dihedrals(coords[4],  coords[5], coords[6], coords[7])

        self.assertEquals(ang2, ang[0])

        fd = FiniteDifferenceDerivative(
            lambda pt: pts_dihedrals(pt[..., 0, :], pt[..., 1, :], pt[..., 2, :], pt[..., 3, :]),
            function_shape=((None, 4, 3), 0),
            mesh_spacing=1.0e-5
        )
        dihedDeriv_fd = FiniteDifferenceDerivative(
            lambda pts: dihed_deriv(pts, 0, 1, 2, 3, order=1)[1].squeeze().transpose((1, 0, 2)),
            function_shape=((None, 4, 3), (None, 4, 3)),
            mesh_spacing=1.0e-5
        )

        fd1, fd2 = fd(coords[(4, 5, 6, 7),]).derivative_tensor([1, 2])
        fd2_22 = dihedDeriv_fd(coords[(4, 5, 6, 7),]).derivative_tensor(1)

        self.assertTrue(np.allclose(deriv.flatten(), fd1.flatten()), msg="{} and {} aren't close".format(
            deriv.flatten(), fd1.flatten()
        ))

        d2_flat = np.concatenate(
            [
                np.concatenate([deriv_2[0, 0], deriv_2[0, 1], deriv_2[0, 2], deriv_2[0, 3]], axis=1),
                np.concatenate([deriv_2[1, 0], deriv_2[1, 1], deriv_2[1, 2], deriv_2[1, 3]], axis=1),
                np.concatenate([deriv_2[2, 0], deriv_2[2, 1], deriv_2[2, 2], deriv_2[2, 3]], axis=1),
                np.concatenate([deriv_2[3, 0], deriv_2[3, 1], deriv_2[3, 2], deriv_2[3, 3]], axis=1)
            ],
            axis=0
        )

        bleh = fd2_22.reshape(12, 12)
        # raise Exception("\n"+"\n".join("{} {}".format(a, b) for a, b in zip(
        #     np.round(deriv_2[2, 2], 3), np.round(bleh[6:9, 6:9], 3))
        #                                ))
        # raise Exception(np.round(d2_flat-bleh, 3))
        # raise Exception("\n"+"\n".join("{}\n{}".format(a, b) for a, b in zip(np.round(d2_flat, 3), np.round(bleh, 3))))
        self.assertTrue(np.allclose(d2_flat.flatten(), bleh.flatten(), atol=1.0e-7), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), bleh.flatten()
        ))
        self.assertTrue(np.allclose(d2_flat.flatten(), fd2.flatten(), atol=1.0e-3), msg="d2: {} and {} differ".format(
            d2_flat.flatten(), fd2.flatten()
        ))
```

 </div>
</div>
___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Numputils/AnalyticDerivs/dihed_deriv.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Numputils/AnalyticDerivs/dihed_deriv.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Numputils/AnalyticDerivs/dihed_deriv.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Numputils/AnalyticDerivs/dihed_deriv.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Numputils/AnalyticDerivs.py?message=Update%20Docs)