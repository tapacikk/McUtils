## <a id="McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.ZMatrixCoordinateSystem">ZMatrixCoordinateSystem</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L98)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L98?message=Update%20Docs)]
</div>

Represents ZMatrix coordinates generally

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
name: str
```
<a id="McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.ZMatrixCoordinateSystem.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, converter_options=None, dimension=(None, None), coordinate_shape=(None, 3), **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L103)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L103?message=Update%20Docs)]
</div>


- `converter_options`: `None | dict`
    >options to be passed through to a `CoordinateSystemConverter`
- `coordinate_shape`: `Iterable[None | int]`
    >shape of a single coordinate in this coordiante system
- `dimension`: `Iterable[None | int]`
    >the dimension of the coordinate system
- `opts`: `Any`
    >other options, if `converter_options` is None, these are used as the `converter_options`

<a id="McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.ZMatrixCoordinateSystem.jacobian_prep_coordinates" class="docs-object-method">&nbsp;</a> 
```python
jacobian_prep_coordinates(self, coord, displacements, values, dihedral_cutoff=4): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L122)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L122?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [CartesianToZMatrixJacobian](#CartesianToZMatrixJacobian)
- [CH5ZMatJacobian](#CH5ZMatJacobian)

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
from Peeves.Timer import Timer
from unittest import TestCase
from McUtils.Coordinerds import *
from McUtils.Plots import *
from McUtils.Numputils import *
import sys, numpy as np
```

All tests are wrapped in a test class
```python
class ConverterTest(TestCase):
    def setUp(self):
        super().setUp()
        self.initialize_data()
        self.load()
    def loaded(self):
        return hasattr(self, "cases")
    def load(self, n=10):
        if not self.loaded:
            self.cases = n
            self.transforms = DataGenerator.mats(n)
            self.shifts = DataGenerator.vecs(n)
            self.mats = affine_matrix(self.transforms, self.shifts)
    def initialize_data(self):
        self.n = 10
        self.test_zmats = CoordinateSet(DataGenerator.zmats(self.n, 15), system=ZMatrixCoordinates)
        self.test_carts = CoordinateSet(DataGenerator.multicoords(self.n, 10))
        self.test_structure = [
            [ 0.0,                    0.0,                   0.0                ],
            [ 0.5312106220949451,     0.0,                   0.0                ],
            [ 5.4908987527698905e-2,  0.5746865893353914,    0.0                ],
            [-6.188515885294378e-2,  -2.4189926062338385e-2, 0.4721688095375285 ],
            [ 1.53308938205413e-2,    0.3833690190410768,    0.23086294551212294],
            [ 0.1310095622893345,     0.30435650497612,      0.5316931774973834 ]
        ]
        self.dihed_test_structure = np.array([
            [0.0, 0.0, 0.0 ],
            [-0.8247121421923925, -0.629530611338456, 1.775332267901544 ],
            [0.1318851447521099, 2.088940054609643, 0.0],
            [1.786540362044548, -1.386051328559878, 0.0],
            [2.233806981137821, 0.3567096955165336, 0.0],
            [-0.8247121421923925, -0.629530611338456, -1.775332267901544]
        ])
        self.zm_conv_test_structure = np.array([
            [1.0, 0.0, 1.0],
            [-0.8247121421923925, -0.629530611338456, 1.775332267901544],
            [0.1318851447521099, 2.088940054609643, 0.0],
            [1.786540362044548, -1.386051328559878, 0.0],
            [2.233806981137821, 0.3567096955165336, 0.0],
            [-0.8247121421923925, -0.629530611338456, -1.775332267901544]
        ])
```

 </div>
</div>

#### <a name="CartesianToZMatrixJacobian">CartesianToZMatrixJacobian</a>
```python
    def test_CartesianToZMatrixJacobian(self):
        n = 10
        test_coords = DataGenerator.coords(n)
        # test_coords = np.array([[0, 0, 0], [1, 1, 0], [1, 2, 0], [0, 2, 1], [0, -2, -1]])
        coord_set = CoordinateSet(test_coords)
        # ordr = [
        #           [0, -1, -1, -1],
        #           [1,  0, -1, -1],
        #           [2,  0,  1, -1],
        #           [3,  1,  2,  0],
        #           [4,  0,  3,  2]
        #           ]

        icrds = coord_set.convert(ZMatrixCoordinates)#, ordering=ordr)
        # print(icrds)
        # wat = icrds.convert(CartesianCoordinates3D)

        internals = ZMatrixCoordinateSystem(**icrds.converter_options)
        ijacob = icrds.jacobian(CartesianCoordinates3D).reshape((n - 1) * 3, n * 3)
        nijacob = icrds.jacobian(CartesianCoordinates3D, all_numerical=True, stencil=3).reshape((n-1)*3, n*3)
        jacob = coord_set.jacobian(internals, stencil=3).reshape(n * 3, (n - 1) * 3)
        njacob = coord_set.jacobian(internals, all_numerical=True).reshape(n*3, (n-1)*3)
        # with Timer("Block Z2C"):
        #     wat = icrds.convert(CartesianCoordinates3D)
        # with Timer("Block Z2C Analytic"):
        #     ijacob = icrds.jacobian(CartesianCoordinates3D).reshape((n-1)*3, n*3)
        # with Timer("Block Z2C Numerical"):
        #     nijacob = icrds.jacobian(CartesianCoordinates3D, all_numerical=True, stencil=3).reshape((n-1)*3, n*3)
        # with Timer("Block C2Z"):
        #     icrds = coord_set.convert(ZMatrixCoordinates)#, ordering=ordr)
        # with Timer("Block C2Z Analytic"):
        #     jacob = coord_set.jacobian(internals, stencil=3).reshape(n * 3, (n - 1) * 3)
        # with Timer("Block C2Z Numerical"):
        #     njacob = coord_set.jacobian(internals, all_numerical=True).reshape(n*3, (n-1)*3)
        # raise Exception("wat")
        # ordcrd = coord_set[np.array(ordr, int)[:, 0]]
        # raise Exception(ordcrd-wat)

        # g = GraphicsGrid(ncols=3, nrows=1, image_size=(900, 600))
        # ArrayPlot(jacob, figure=g[0, 0])
        # ArrayPlot(njacob, figure=g[0, 1])
        # ArrayPlot(np.round(njacob-jacob, 4), figure=g[0, 2])
        # g.padding=.05
        # g.padding_top=.5
        # # g.padding_bottom=0
        # g.show()

        # g = GraphicsGrid(ncols=3, nrows=2, image_size=(900, 600))
        # ArrayPlot(jacob,          figure=g[0, 0])
        # ArrayPlot(njacob,         figure=g[1, 0])
        # ArrayPlot(jacob - njacob, figure=g[0, 1])
        # ArrayPlot(ijacob,         figure=g[1, 1])
        # ArrayPlot(nijacob@jacob,  figure=g[0, 2])
        # ArrayPlot(ijacob@jacob,   figure=g[1, 2])
        # g.show()

        self.assertTrue(np.allclose(jacob,  njacob), msg="{} too large".format(np.sum(np.abs(jacob-njacob))))
        self.assertTrue(np.allclose(ijacob,  nijacob))
        self.assertEquals(jacob.shape, (n*3, (n-1)*3)) # we always lose one atom
        self.assertAlmostEqual(np.sum((ijacob@jacob)), 3*n-6, 3)
```
#### <a name="CH5ZMatJacobian">CH5ZMatJacobian</a>
```python
    def test_CH5ZMatJacobian(self):
        coord_set = CoordinateSet([
            [
                [ 0.000000000000000,    0.000000000000000,  0.000000000000000],
                [ 0.1318851447521099,   2.088940054609643,  0.000000000000000],
                [ 1.786540362044548,   -1.386051328559878,  0.000000000000000],
                [ 2.233806981137821,    0.3567096955165336, 0.000000000000000],
                [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                [-0.8247121421923925, -0.6295306113384560,  1.775332267901544]
                ]
            ]*100,
            system=CartesianCoordinates3D
        )

        zmat_system = ZMatrixCoordinateSystem(
            ordering=[
                [0,  0, -1, -1],
                [1,  0,  1, -1],
                [2,  0,  1,  2],
                [3,  0,  1,  2],
                [4,  0,  1,  2],
                [5,  0,  1,  2]
            ]
        )
        # zmcs = coord_set.convert(ZMatrixCoordinates, ordering=zmat_ordering)

        jacob = coord_set.jacobian(
            zmat_system,
            stencil=5,
            prep=lambda coord, disps, zmcs: (disps, zmcs[..., :, 1]),
            all_numerical = True
        )
        self.assertEquals(jacob.shape, (np.product(coord_set.shape[1:]), 100, 5)) # I requested 5 bond lengths

        # the analytic derivs. track a slightly different shape
        jacob = coord_set.jacobian(
            zmat_system,
            stencil=5,
            prep=lambda coord, disps, zmcs: (disps, zmcs[..., :, 1])
        )
        self.assertEquals(jacob.shape, (100,) + coord_set.shape[1:] + (5, 3))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/ZMatrixCoordinateSystem.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/ZMatrixCoordinateSystem.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/ZMatrixCoordinateSystem.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/ZMatrixCoordinateSystem.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L98?message=Update%20Docs)