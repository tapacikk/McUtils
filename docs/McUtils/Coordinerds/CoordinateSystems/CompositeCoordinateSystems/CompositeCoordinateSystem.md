## <a id="McUtils.Coordinerds.CoordinateSystems.CompositeCoordinateSystems.CompositeCoordinateSystem">CompositeCoordinateSystem</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L14)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L14?message=Update%20Docs)]
</div>

Defines a coordinate system that comes from applying a transformation
to another coordinate system

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Coordinerds.CoordinateSystems.CompositeCoordinateSystems.CompositeCoordinateSystem.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, base_system, conversion, inverse_conversion=None, name=None, batched=None, pointwise=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L21)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L21?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CompositeCoordinateSystems.CompositeCoordinateSystem.canonical_name" class="docs-object-method">&nbsp;</a> 
```python
canonical_name(name, conversion): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L29)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L29?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CompositeCoordinateSystems.CompositeCoordinateSystem.register" class="docs-object-method">&nbsp;</a> 
```python
register(base_system, conversion, inverse_conversion=None, name=None, batched=None, pointwise=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L47)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L47?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CompositeCoordinateSystems.CompositeCoordinateSystem.unregister" class="docs-object-method">&nbsp;</a> 
```python
unregister(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L57)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L57?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CompositeCoordinateSystems.CompositeCoordinateSystem.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L59)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L59?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [CustomConversion](#CustomConversion)
- [ChainCustomConversion](#ChainCustomConversion)

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

#### <a name="CustomConversion">CustomConversion</a>
```python
    def test_CustomConversion(self):

        def invert(coords, **opts):
            return -coords, opts

        new = CompositeCoordinateSystem.register(CartesianCoordinates3D, invert, pointwise=False, inverse_conversion=invert)
        coord_set = CoordinateSet(DataGenerator.multicoords(5, 10))
        crds = coord_set.convert(new)
        old = crds.convert(coord_set.system)

        self.assertEquals(np.sum(crds + coord_set), 0.)
        self.assertEquals(np.sum(coord_set - old), 0.)

        self.assertAlmostEquals(np.sum(coord_set.jacobian(new)[:, 0].reshape(30, 30) + np.eye(30, 30)), 0.)
```
#### <a name="ChainCustomConversion">ChainCustomConversion</a>
```python
    def test_ChainCustomConversion(self):
        def invert(coords, **opts):
            return -coords, opts

        new = CompositeCoordinateSystem.register(SphericalCoordinates, invert, pointwise=False, inverse_conversion=invert)
        coord_set = CoordinateSet(DataGenerator.multicoords(5, 10))
        crds = coord_set.convert(new)
        old = crds.convert(coord_set.system)

        self.assertAlmostEqual(np.sum(coord_set - old)[()], 0.)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateSystems/CompositeCoordinateSystems/CompositeCoordinateSystem.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateSystems/CompositeCoordinateSystems/CompositeCoordinateSystem.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateSystems/CompositeCoordinateSystems/CompositeCoordinateSystem.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateSystems/CompositeCoordinateSystems/CompositeCoordinateSystem.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CompositeCoordinateSystems.py#L14?message=Update%20Docs)