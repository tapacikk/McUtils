## <a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet">CoordinateSet</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L20)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L20?message=Update%20Docs)]
</div>

A subclass of np.ndarray that lives in an explicit coordinate system and can convert between them

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.__new__" class="docs-object-method">&nbsp;</a> 
```python
__new__(cls, coords, system=CoordinateSystem(Cartesian3D, dimension=(None, 3), matrix=None), converter_options=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L26)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L26?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, coords, system=CoordinateSystem(Cartesian3D, dimension=(None, 3), matrix=None), converter_options=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L32)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L32?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.__array_finalize__" class="docs-object-method">&nbsp;</a> 
```python
__array_finalize__(self, coords): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L37)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L37?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.__str__" class="docs-object-method">&nbsp;</a> 
```python
__str__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L62)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L62?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.__eq__" class="docs-object-method">&nbsp;</a> 
```python
__eq__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L65)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L65?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.multiconfig" class="docs-object-method">&nbsp;</a> 
```python
@property
multiconfig(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L?message=Update%20Docs)]
</div>

Determines whether self.coords represents multiple configurations of the coordinates
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.transform" class="docs-object-method">&nbsp;</a> 
```python
transform(self, tf): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L80)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L80?message=Update%20Docs)]
</div>

Applies a transformation to the stored coordinates
- `tf`: `Any`
    >the transformation function to apply to the system
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, system, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L92)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L92?message=Update%20Docs)]
</div>

Converts across coordinate systems
- `system`: `CoordinateSystem`
    >the target coordinate system
- `:returns`: `CoordinateSet`
    >new_coords

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.derivatives" class="docs-object-method">&nbsp;</a> 
```python
derivatives(self, function, order=1, coordinates=None, result_shape=None, **fd_options): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L112)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L112?message=Update%20Docs)]
</div>

Takes derivatives of `function` with respect to the current geometry
- `function`: `Any`
    >No description...
- `order`: `Any`
    >No description...
- `coordinates`: `Any`
    >No description...
- `fd_options`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSet.CoordinateSet.jacobian" class="docs-object-method">&nbsp;</a> 
```python
jacobian(self, system, order=1, coordinates=None, converter_options=None, all_numerical=False, analytic_deriv_order=None, **fd_options): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L136)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L136?message=Update%20Docs)]
</div>

Delegates to the jacobian function of the current coordinate system
- `system`: `Any`
    >No description...
- `order`: `Any`
    >No description...
- `mesh_spacing`: `Any`
    >No description...
- `prep`: `Any`
    >No description...
- `coordinates`: `Any`
    >No description...
- `fd_options`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [CoordinateSet](#CoordinateSet)
- [CartesianToZMatrix](#CartesianToZMatrix)
- [CartesianToZMatrixMulti](#CartesianToZMatrixMulti)
- [CartesianToZMatrixAndBack](#CartesianToZMatrixAndBack)
- [ExpansionCoordinates](#ExpansionCoordinates)
- [CartesianToZMatrixJacobian](#CartesianToZMatrixJacobian)
- [CartesianToZMatrixMultiJacobian](#CartesianToZMatrixMultiJacobian)
- [CH5ZMatJacobian](#CH5ZMatJacobian)
- [CartesianToZMatrixJacobian2](#CartesianToZMatrixJacobian2)
- [CartesianToZMatrixJacobian2Planar](#CartesianToZMatrixJacobian2Planar)
- [CartesianToZMatrixMultiJacobian2](#CartesianToZMatrixMultiJacobian2)
- [CartesianToZMatrixMultiJacobian1](#CartesianToZMatrixMultiJacobian1)
- [CartesianToZMatrixMultiJacobian10](#CartesianToZMatrixMultiJacobian10)
- [CartesianToZMatrixMultiJacobian2Timing10](#CartesianToZMatrixMultiJacobian2Timing10)
- [CartesianToZMatrixMultiJacobian3Timing1](#CartesianToZMatrixMultiJacobian3Timing1)
- [CartesianToZMatrixMultiJacobian3Timing10](#CartesianToZMatrixMultiJacobian3Timing10)
- [CartesianToZMatrixMultiJacobian3](#CartesianToZMatrixMultiJacobian3)
- [CartesianToZMatrixMultiJacobianTargeted](#CartesianToZMatrixMultiJacobianTargeted)

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

#### <a name="CoordinateSet">CoordinateSet</a>
```python
    def test_CoordinateSet(self):
        import numpy as np
        coord_set = CoordinateSet(DataGenerator.coords(500))
        self.assertIsInstance(coord_set, np.ndarray)
```
#### <a name="CartesianToZMatrix">CartesianToZMatrix</a>
```python
    def test_CartesianToZMatrix(self):
        coord_set = CoordinateSet(DataGenerator.coords(10))
        coord_set = coord_set.convert(ZMatrixCoordinates, use_rad = False)
        self.assertEqual(coord_set.shape, (9, 3))
```
#### <a name="CartesianToZMatrixMulti">CartesianToZMatrixMulti</a>
```python
    def test_CartesianToZMatrixMulti(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        coord_set = coord_set.convert(ZMatrixCoordinates, use_rad = False)
        self.assertEqual(coord_set.shape, (10, 9, 3))
```
#### <a name="CartesianToZMatrixAndBack">CartesianToZMatrixAndBack</a>
```python
    def test_CartesianToZMatrixAndBack(self):
        cs1 = coord_set = CoordinateSet([self.zm_conv_test_structure]*4, CartesianCoordinates3D)
        coord_set = coord_set.convert(ZMatrixCoordinates)
        coord_set = coord_set.convert(CartesianCoordinates3D)
        cs2 = coord_set
        self.assertEqual(round(np.linalg.norm(cs2 - cs1), 8), 0.)
```
#### <a name="ExpansionCoordinates">ExpansionCoordinates</a>
```python
    def test_ExpansionCoordinates(self):
        np.random.seed(0)
        coord_set = CoordinateSet([self.test_structure] * 2)
        expansion_1 = np.random.rand(len(self.test_structure)*3, len(self.test_structure)*3)
        expansion_1 = (expansion_1 / np.broadcast_to(np.linalg.norm(expansion_1, axis=0), expansion_1.shape))
        cs1 = CoordinateSystem("CartesianExpanded",
                               basis=CartesianCoordinates3D,
                               matrix=expansion_1
                               )
        expansion_2 = np.random.rand((len(self.test_structure) - 1) * 3, (len(self.test_structure) - 1) * 3)
        expansion_2 = (expansion_2 / np.broadcast_to(np.linalg.norm(expansion_2, axis=0), expansion_2.shape))
        cs2 = CoordinateSystem("ZMatrixExpanded",
                               basis=ZMatrixCoordinates,
                               matrix=expansion_2
                               )
        coord_set2 = coord_set.convert(cs1)
        coord_set2 = coord_set2.convert(cs2)
        coord_set2 = coord_set2.convert(CartesianCoordinates3D)
        self.assertEqual(round(np.linalg.norm(coord_set2 - coord_set), 8), 0.)
```
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
#### <a name="CartesianToZMatrixMultiJacobian">CartesianToZMatrixMultiJacobian</a>
```python
    def test_CartesianToZMatrixMultiJacobian(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, stencil = 5)
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (10, 10, 3, 10 - 1, 3 ))
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
#### <a name="CartesianToZMatrixJacobian2">CartesianToZMatrixJacobian2</a>
```python
    def test_CartesianToZMatrixJacobian2(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10)[0])
        njacob = coord_set.jacobian(ZMatrixCoordinates, 2, stencil=5, all_numerical=True)
        self.assertEquals(njacob.shape, (10 * 3, 10 * 3, 10 - 1, 3))  # we always lose one atom

        jacob = coord_set.jacobian(ZMatrixCoordinates, 2, stencil=5)  # semi-analytic
        self.assertEquals(jacob.shape, (10, 3, 10, 3, 10 - 1, 3))  # we always lose one atom

        # jacob = jacob.reshape((10, 3, 10, 3, 10 - 1, 3))

        # import McUtils.Plots as plt
        #
        # bleh_a = np.round(np.reshape(njacob, (900, 27)), 8)
        # bleh_b = np.round(np.reshape(jacob, (900, 27)), 8)
        # bleh_c = np.round(bleh_a - bleh_b, 10)
        # bleh_bleh = np.where(bleh_c != 0.)
        #
        # plt.ArrayPlot( ( bleh_a == 0. ).astype(int) )
        # plt.ArrayPlot( ( bleh_b == 0. ).astype(int) )
        # plt.ArrayPlot( ( np.round(bleh_c, 5) == 0. ).astype(int) ).show()

        njacob = njacob.reshape((10, 3, 10, 3, 10 - 1, 3))
        diffs = njacob - jacob
        bleh_bleh = np.where(np.round(diffs, 3) != 0.)
        # print(bleh_bleh)

        # print("???", np.round(jacob[0, :, 0, :, :, 0], 2), np.round(njacob[0, :, 0, :, :, 2], 2))
        self.assertTrue(
            np.allclose(diffs, 0., atol=1.0e-3),
            msg="wat: {}".format(np.max(np.abs(np.round(diffs, 3))))
        )
```
#### <a name="CartesianToZMatrixJacobian2Planar">CartesianToZMatrixJacobian2Planar</a>
```python
    def test_CartesianToZMatrixJacobian2Planar(self):

        coord_set = CoordinateSet( # water
            np.array(
                [[-9.84847483e-18, -1.38777878e-17, 9.91295048e-02],
                 [-9.84847483e-18, -1.38777878e-17, 1.09912950e+00],
                 [ 1.00000000e+00, 9.71445147e-17, 9.91295048e-02],
                 [ 2.46519033e-32, -1.38777878e-17, 2.25076602e-01],
                 [-1.97215226e-31, 1.43714410e+00, -9.00306410e-01],
                 [-1.75999392e-16, -1.43714410e+00, -9.00306410e-01]]
            )
        )
        coord_set_2 = CoordinateSet( # HOD
            np.array([
                [-1.86403557e-17, -7.60465240e-02,  4.62443228e-02],
                [ 6.70904773e-17, -7.60465240e-02, -9.53755677e-01],
                [ 9.29682337e-01,  2.92315732e-01,  4.62443228e-02],
                [ 2.46519033e-32, -1.38777878e-17,  2.25076602e-01],
                [-1.97215226e-31,  1.43714410e+00, -9.00306410e-01],
                [-1.75999392e-16, -1.43714410e+00, -9.00306410e-01]
            ])
        )
        internal_ordering = [
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  0,  1, -1],
            [3,  0,  2,  1],
            [4,  3,  1,  2],
            [5,  3,  4,  1]
        ]
        coord_set.converter_options = {'ordering': internal_ordering}
        njacob = coord_set.jacobian(ZMatrixCoordinates, 2, stencil=5, analytic_deriv_order=1) # only do analytic firsts
        self.assertEquals(njacob.shape, (6 * 3, 6, 3, 6 - 1, 3))  # we always lose one atom

        jacob = coord_set.jacobian(ZMatrixCoordinates, 2, stencil=5)  # totally-analytic
        self.assertEquals(jacob.shape, (6, 3, 6, 3, 6 - 1, 3))  # we always lose one atom

        njacob = njacob.reshape((6, 3, 6, 3, 6 - 1, 3))

        diffs = njacob - jacob
        ehhh = np.round(diffs, 3)
        # print(
        #     njacob[ np.abs(ehhh) > 0 ],
        #     jacob[np.abs(ehhh) > 0],
        # )

        print(
            np.array(np.where(np.abs(jacob) > 100)).T
        )
        print(np.max(np.abs(njacob)), np.max(np.abs(jacob)))
        self.assertTrue(
            np.allclose(diffs, 0., atol=1.0e-4),
            msg="wat: {}".format(np.max(np.abs(np.round(diffs, 6))))
        )
```
#### <a name="CartesianToZMatrixMultiJacobian2">CartesianToZMatrixMultiJacobian2</a>
```python
    def test_CartesianToZMatrixMultiJacobian2(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        njacob = coord_set.jacobian(ZMatrixCoordinates, 2, stencil = 5, all_numerical=True)
        # TensorPlot(jacob[0],
        #            ticks_style = [
        #                dict(
        #                    bottom = False,
        #                    top = False,
        #                    labelbottom = False
        #                ),
        #                dict(
        #                    right = False,
        #                    left = False,
        #                    labelleft = False
        #                )
        #            ]
        #            ).show()
        self.assertEquals(njacob.shape, (10*3, 10*3, 10, 10 - 1, 3)) # we always lose one atom


        jacob = coord_set.jacobian(ZMatrixCoordinates, 2, stencil = 5) # semi-analytic
        self.assertEquals(jacob.shape, (10, 10, 3, 10, 3, 10 - 1, 3))
```
#### <a name="CartesianToZMatrixMultiJacobian1">CartesianToZMatrixMultiJacobian1</a>
```python
    def test_CartesianToZMatrixMultiJacobian1(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(1, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, stencil = 5)
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (1, 10*3, 10 * 3 - 3 ))
```
#### <a name="CartesianToZMatrixMultiJacobian10">CartesianToZMatrixMultiJacobian10</a>
```python
    def test_CartesianToZMatrixMultiJacobian10(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, stencil = 5)
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (10, 10*3, 10 * 3 - 3 ))
```
#### <a name="CartesianToZMatrixMultiJacobian2Timing10">CartesianToZMatrixMultiJacobian2Timing10</a>
```python
    def test_CartesianToZMatrixMultiJacobian2Timing10(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, 2, stencil = 5)
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (10, 10*3, 10*3, 10 * 3 - 3 ))
```
#### <a name="CartesianToZMatrixMultiJacobian3Timing1">CartesianToZMatrixMultiJacobian3Timing1</a>
```python
    def test_CartesianToZMatrixMultiJacobian3Timing1(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(1, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, 3, stencil = 5)
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (1, 10*3, 10*3, 10*3, 10 * 3 - 3 ))
```
#### <a name="CartesianToZMatrixMultiJacobian3Timing10">CartesianToZMatrixMultiJacobian3Timing10</a>
```python
    def test_CartesianToZMatrixMultiJacobian3Timing10(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, 3, stencil = 5)
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (10, 10*3, 10*3, 10*3, 10 * 3 - 3 ))
```
#### <a name="CartesianToZMatrixMultiJacobian3">CartesianToZMatrixMultiJacobian3</a>
```python
    def test_CartesianToZMatrixMultiJacobian3(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, 3, stencil=5)
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (10 * 3, 10 * 3, 10, 10, 3, 10 - 1, 3))
```
#### <a name="CartesianToZMatrixMultiJacobianTargeted">CartesianToZMatrixMultiJacobianTargeted</a>
```python
    def test_CartesianToZMatrixMultiJacobianTargeted(self):
        coord_set = CoordinateSet(DataGenerator.multicoords(10, 10))
        jacob = coord_set.jacobian(ZMatrixCoordinates, stencil=5, coordinates=[[1, 2, 3], None])
        # ArrayPlot(jacob[0], colorbar=True).show()
        self.assertEquals(jacob.shape, (10, 10, 3, 10 - 1, 3))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateSystems/CoordinateSet/CoordinateSet.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateSystems/CoordinateSet/CoordinateSet.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateSystems/CoordinateSet/CoordinateSet.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateSystems/CoordinateSet/CoordinateSet.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CoordinateSet.py#L20?message=Update%20Docs)