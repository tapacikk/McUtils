## <a id="McUtils.Data.ConstantsData.UnitsDataHandler">UnitsDataHandler</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/ConstantsData.py#L60)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/ConstantsData.py#L60?message=Update%20Docs)]
</div>

A DataHandler that's built for use with the units data we've collected.
Usually used through the `UnitsData` object.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
prefix_map: OrderedDict
postfix_map: OrderedDict
```
<a id="McUtils.Data.ConstantsData.UnitsDataHandler.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/ConstantsData.py#L84)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/ConstantsData.py#L84?message=Update%20Docs)]
</div>

<a id="McUtils.Data.ConstantsData.UnitsDataHandler.load" class="docs-object-method">&nbsp;</a> 
```python
load(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/ConstantsData.py#L88)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/ConstantsData.py#L88?message=Update%20Docs)]
</div>

<a id="McUtils.Data.ConstantsData.UnitsDataHandler.find_conversion" class="docs-object-method">&nbsp;</a> 
```python
find_conversion(self, unit, target): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/ConstantsData.py#L368)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/ConstantsData.py#L368?message=Update%20Docs)]
</div>

Attempts to find a conversion between two sets of units. Currently only implemented for "plain" units.
- `unit`: `Any`
    >No description...
- `target`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Data.ConstantsData.UnitsDataHandler.add_conversion" class="docs-object-method">&nbsp;</a> 
```python
add_conversion(self, unit, target, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/ConstantsData.py#L391)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/ConstantsData.py#L391?message=Update%20Docs)]
</div>

<a id="McUtils.Data.ConstantsData.UnitsDataHandler.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, unit, target): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Data/ConstantsData.py#L395)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Data/ConstantsData.py#L395?message=Update%20Docs)]
</div>

Converts base unit into target using the scraped NIST data
- `unit`: `Any`
    >No description...
- `target`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Data/ConstantsData/UnitsDataHandler.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Data/ConstantsData/UnitsDataHandler.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Data/ConstantsData/UnitsDataHandler.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Data/ConstantsData/UnitsDataHandler.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Data/ConstantsData.py#L60?message=Update%20Docs)