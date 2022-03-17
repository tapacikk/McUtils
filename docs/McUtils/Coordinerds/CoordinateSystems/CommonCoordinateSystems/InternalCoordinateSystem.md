## <a id="McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.InternalCoordinateSystem">InternalCoordinateSystem</a>
Represents Internal coordinates generally

### Properties and Methods
```python
name: str
```
<a id="McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.InternalCoordinateSystem.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, dimension=None, coordinate_shape=None, converter_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L50)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L50?message=Update%20Docs)]
</div>


- `converter_options`: `None | dict`
    >options to be passed through to a `CoordinateSystemConverter`
- `coordinate_shape`: `Iterable[None | int]`
    >shape of a single coordinate in this coordiante system
- `dimension`: `Iterable[None | int]`
    >the dimension of the coordinate system
- `opts`: `Any`
    >other options, if `converter_options` is None, these are used as the `converter_options`





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py?message=Update%20Docs)