## <a id="McUtils.McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.ZMatrixCoordinateSystem">ZMatrixCoordinateSystem</a>
Represents ZMatrix coordinates generally

### Properties and Methods
```python
name: str
```
<a id="McUtils.McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.ZMatrixCoordinateSystem.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, converter_options=None, dimension=(None, None), coordinate_shape=(None, 3), **opts): 
```

- `converter_options`: `None | dict`
    >options to be passed through to a `CoordinateSystemConverter`
- `coordinate_shape`: `Iterable[None | int]`
    >shape of a single coordinate in this coordiante system
- `dimension`: `Iterable[None | int]`
    >the dimension of the coordinate system
- `opts`: `Any`
    >other options, if `converter_options` is None, these are used as the `converter_options`

### Examples


