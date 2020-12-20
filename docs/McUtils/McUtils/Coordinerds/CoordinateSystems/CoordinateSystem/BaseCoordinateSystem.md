## <a id="McUtils.McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.BaseCoordinateSystem">BaseCoordinateSystem</a>
A CoordinateSystem object that can't be reduced further.
A common choice might be Cartesian coordinates or internal coordinates.
This allows us to define flexible `CoordinateSystem` subclasses that we _don't_ expect to be used as a base

### Properties and Methods
<a id="McUtils.McUtils.Coordinerds.CoordinateSystems.CoordinateSystem.BaseCoordinateSystem.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, name, dimension=None, matrix=None, coordinate_shape=None, converter_options=None): 
```

### Examples
