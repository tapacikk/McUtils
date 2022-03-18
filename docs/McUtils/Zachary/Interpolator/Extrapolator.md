## <a id="McUtils.Zachary.Interpolator.Extrapolator">Extrapolator</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator.py#L458)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L458?message=Update%20Docs)]
</div>

A general purpose that takes your data and just extrapolates it.
This currently only exists in template format.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.Interpolator.Extrapolator.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, extrapolation_function, warning=False, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator.py#L463)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L463?message=Update%20Docs)]
</div>


- `extrapolation_function`: `None | function | Callable | Interpolator`
    >the function to handle extrapolation off the interpolation grid
- `warning`: `bool`
    >whether to emit a message warning about extrapolation occurring
- `opts`: `Any`
    >the options to feed into the extrapolator call

<a id="McUtils.Zachary.Interpolator.Extrapolator.derivative" class="docs-object-method">&nbsp;</a> 
```python
derivative(self, n): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator.py#L480)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L480?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Interpolator.Extrapolator.find_extrapolated_points" class="docs-object-method">&nbsp;</a> 
```python
find_extrapolated_points(self, gps, vals, extrap_value=nan): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator.py#L486)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L486?message=Update%20Docs)]
</div>

Currently super rough heuristics to determine at which points we need to extrapolate
- `gps`: `Any`
    >No description...
- `vals`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Interpolator.Extrapolator.apply" class="docs-object-method">&nbsp;</a> 
```python
apply(self, gps, vals, extrap_value=nan): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator.py#L507)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L507?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Interpolator.Extrapolator.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator.py#L515)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L515?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Interpolator/Extrapolator.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Interpolator/Extrapolator.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Interpolator/Extrapolator.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Interpolator/Extrapolator.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L458?message=Update%20Docs)