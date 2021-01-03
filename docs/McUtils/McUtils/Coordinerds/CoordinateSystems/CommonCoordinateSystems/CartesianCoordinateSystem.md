## <a id="McUtils.McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.CartesianCoordinateSystem">CartesianCoordinateSystem</a>
Represents Cartesian coordinates generally

### Properties and Methods
```python
name: str
```
<a id="McUtils.McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.CartesianCoordinateSystem.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, dimension=None, converter_options=None, **opts): 
```

- `converter_options`: `None | dict`
    >options to be passed through to a `CoordinateSystemConverter`
- `dimension`: `Iterable[None | int]`
    >the dimension of the coordinate system
- `opts`: `Any`
    >other options, if `converter_options` is None, these are used as the `converter_options`

### Examples


