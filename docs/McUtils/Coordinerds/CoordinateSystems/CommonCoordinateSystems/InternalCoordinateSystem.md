## <a id="McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.InternalCoordinateSystem">InternalCoordinateSystem</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L44)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L44?message=Update%20Docs)]
</div>

Represents Internal coordinates generally

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
name: str
```
<a id="McUtils.Coordinerds.CoordinateSystems.CommonCoordinateSystems.InternalCoordinateSystem.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, dimension=None, coordinate_shape=None, converter_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L50)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L50?message=Update%20Docs)]
</div>


- `converter_options`: `None | dict`
    >options to be passed through to a `CoordinateSystemConverter`
- `coordinate_shape`: `Iterable[None | int]`
    >shape of a single coordinate in this coordiante system
- `dimension`: `Iterable[None | int]`
    >the dimension of the coordinate system
- `opts`: `Any`
    >other options, if `converter_options` is None, these are used as the `converter_options`

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateSystems/CommonCoordinateSystems.py#L44?message=Update%20Docs)