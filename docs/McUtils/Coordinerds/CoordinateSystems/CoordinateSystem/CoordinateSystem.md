## <a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem">CoordinateSystem</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L19)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L19?message=Update%20Docs)]
</div>

A representation of a coordinate system. It doesn't do much on its own but it *does* provide a way
to unify internal, cartesian, derived type coordinates

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
return_derivs_key: str
```
<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, name=None, basis=None, matrix=None, inverse=None, dimension=None, origin=None, coordinate_shape=None, jacobian_prep=None, converter_options=None, **extra): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L24)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L24?message=Update%20Docs)]
</div>

Sets up the CoordinateSystem object
- `name`: `str`
    >a name to give to the coordinate system
- `basis`: `Any`
    >a basis for the coordinate system
- `matrix`: `np.ndarray | None`
    >an expansion coefficient matrix for the set of coordinates in its basis
- `dimension`: `Iterable[None | int]`
    >the dimension of a single configuration in the coordinate system (for validation)
- `jacobian_prep`: `function | None`
    >a function for preparing coordinates to be used in computing the Jacobian
- `coordinate_shape`: `iterable[int]`
    >the actual shape of a single coordinate in the coordinate system

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.pre_convert" class="docs-object-method">&nbsp;</a> 
```python
pre_convert(self, system): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L80)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L80?message=Update%20Docs)]
</div>

A hook to allow for handlign details before converting
- `system`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.basis" class="docs-object-method">&nbsp;</a> 
```python
@property
basis(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L?message=Update%20Docs)]
</div>


- `:returns`: `CoordinateSystem`
    >The basis for the representation of `matrix`

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.origin" class="docs-object-method">&nbsp;</a> 
```python
@property
origin(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L?message=Update%20Docs)]
</div>


- `:returns`: `np.ndarray`
    >The origin for the expansion defined by `matrix`

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.matrix" class="docs-object-method">&nbsp;</a> 
```python
@property
matrix(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L?message=Update%20Docs)]
</div>

The matrix representation in the `CoordinateSystem.basis`
        `None` is shorthand for the identity matrix
- `:returns`: `np.ndarray`
    >mat

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.inverse" class="docs-object-method">&nbsp;</a> 
```python
@property
inverse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L?message=Update%20Docs)]
</div>

The inverse of the representation in the `basis`.
        `None` is shorthand for the inverse or pseudoinverse of `matrix`.
- `:returns`: `np.ndarray`
    >inv

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.dimension" class="docs-object-method">&nbsp;</a> 
```python
@property
dimension(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L?message=Update%20Docs)]
</div>

The dimension of the coordinate system.
        `None` means unspecified dimension
- `:returns`: `int or None`
    >dim

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.converter" class="docs-object-method">&nbsp;</a> 
```python
converter(self, system): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L155)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L155?message=Update%20Docs)]
</div>

Gets the converter from the current system to a new system
- `system`: `CoordinateSystem`
    >the target CoordinateSystem
- `:returns`: `CoordinateSystemConverter`
    >converter object

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.convert_coords" class="docs-object-method">&nbsp;</a> 
```python
convert_coords(self, coords, system, converter=None, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L221)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L221?message=Update%20Docs)]
</div>

Converts coordiantes from the current coordinate system to _system_
- `coords`: `CoordinateSet`
    >No description...
- `system`: `CoordinateSystem`
    >No description...
- `kw`: `Any`
    >options to be passed through to the converter object
- `:returns`: `tuple(np.ndarray, dict)`
    >the converted coordiantes

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.displacement" class="docs-object-method">&nbsp;</a> 
```python
displacement(self, amts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L285)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L285?message=Update%20Docs)]
</div>

Generates a displacement or matrix of displacements based on the vector or matrix amts
        The relevance of this method has become somewhat unclear...
- `amts`: `np.ndarray`
    >No description...
- `:returns`: `np.ndarray`
    >No description...

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.derivatives" class="docs-object-method">&nbsp;</a> 
```python
derivatives(self, coords, function, order=1, coordinates=None, result_shape=None, **finite_difference_options): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L307)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L307?message=Update%20Docs)]
</div>

Computes derivatives for an arbitrary function with respect to this coordinate system.
        Basically a more flexible version of `jacobian`.
- `function`: `Any`
    >No description...
- `order`: `Any`
    >No description...
- `coordinates`: `Any`
    >No description...
- `finite_difference_options`: `Any`
    >No description...
- `:returns`: `np.ndarray`
    >derivative tensor

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.jacobian" class="docs-object-method">&nbsp;</a> 
```python
jacobian(self, coords, system, order=1, coordinates=None, converter_options=None, all_numerical=False, analytic_deriv_order=None, **finite_difference_options): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L421)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L421?message=Update%20Docs)]
</div>

Computes the Jacobian between the current coordinate system and a target coordinate system
- `system`: `CoordinateSystem`
    >the target CoordinateSystem
- `order`: `int | Iterable[int]`
    >the order of the Jacobian to compute, 1 for a standard, 2 for the Hessian, etc.
- `coordinates`: `None | iterable[iterable[int] | None`
    >a spec of which coordinates to generate derivatives for (None means all)
- `mesh_spacing`: `float | np.ndarray`
    >the spacing to use when displacing
- `prep`: `None | function`
    >a function for pre-validating the generated coordinate values and grids
- `fd_options`: `Any`
    >options to be passed straight through to FiniteDifferenceFunction
- `:returns`: `np.ndarray`
    >derivative tensor

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L579)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L579?message=Update%20Docs)]
</div>

Provides a clean representation of a `CoordinateSystem` for printing
- `:returns`: `str`
    >No description...

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.is_compatible" class="docs-object-method">&nbsp;</a> 
```python
is_compatible(self, system): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L586)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L586?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.CoordinateSystem.has_conversion" class="docs-object-method">&nbsp;</a> 
```python
has_conversion(self, system): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L594)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L594?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [ExpansionCoordinates](#ExpansionCoordinates)
- [CartExpanded](#CartExpanded)

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
#### <a name="CartExpanded">CartExpanded</a>
```python
    def test_CartExpanded(self):
        expansion = (np.array(
            [
                [1 / np.sqrt(6), np.sqrt(2 / 3), 1 / np.sqrt(6)],
                [1 / np.sqrt(3), -(1 / np.sqrt(3)), 1 / np.sqrt(3)],
                [1 / np.sqrt(2), 0, -(1 / np.sqrt(2))]
            ]
        ))
        system = CoordinateSystem(
            basis=CartesianCoordinates3D,
            matrix=expansion
        )
        disp = system.displacement(.1)
        self.assertEquals(disp, .1)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateSystems/CoordinateSystem/CoordinateSystem.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateSystems/CoordinateSystem/CoordinateSystem.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateSystems/CoordinateSystem/CoordinateSystem.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateSystems/CoordinateSystem/CoordinateSystem.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CoordinateSystem.py#L19?message=Update%20Docs)