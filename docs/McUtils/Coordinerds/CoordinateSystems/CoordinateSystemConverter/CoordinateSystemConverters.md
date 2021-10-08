## <a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystemConverter.CoordinateSystemConverters">CoordinateSystemConverters</a>
A coordinate converter class. It's a singleton so can't be instantiated.

### Properties and Methods
```python
converters: OrderedDict
converters_dir: str
converters_package: str
```
<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystemConverter.CoordinateSystemConverters.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self): 
```

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystemConverter.CoordinateSystemConverters.get_coordinates" class="docs-object-method">&nbsp;</a>
```python
get_coordinates(coordinate_set): 
```
Extracts coordinates from a coordinate_set

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystemConverter.CoordinateSystemConverters.load_converter" class="docs-object-method">&nbsp;</a>
```python
load_converter(converter): 
```

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystemConverter.CoordinateSystemConverters.get_converter" class="docs-object-method">&nbsp;</a>
```python
get_converter(system1, system2): 
```
Gets the appropriate converter for two CoordinateSystem objects
- `system1`: `CoordinateSystem`
    >No description...
- `system2`: `CoordinateSystem`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateSystems.CoordinateSystemConverter.CoordinateSystemConverters.register_converter" class="docs-object-method">&nbsp;</a>
```python
register_converter(system1, system2, converter): 
```
Registers a converter between two coordinate systems
- `system1`: `CoordinateSystem`
    >No description...
- `system2`: `CoordinateSystem`
    >No description...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Coordinerds/CoordinateSystems/CoordinateSystemConverter/CoordinateSystemConverters.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Coordinerds/CoordinateSystems/CoordinateSystemConverter/CoordinateSystemConverters.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Coordinerds/CoordinateSystems/CoordinateSystemConverter/CoordinateSystemConverters.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Coordinerds/CoordinateSystems/CoordinateSystemConverter/CoordinateSystemConverters.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Coordinerds/CoordinateSystems/CoordinateSystemConverter.py?message=Update%20Docs)