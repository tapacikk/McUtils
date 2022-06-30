## <a id="McUtils.Plots.Styling.ThemeManager">ThemeManager</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L100)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L100?message=Update%20Docs)]
</div>

Simple manager class for plugging into themes in a semi-background agnostic way

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
extra_themes: dict
```
<a id="McUtils.Plots.Styling.ThemeManager.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *theme_names, backend=<Backends.MPL: 'matplotlib'>, graphics_styles=None, **extra_styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L119)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L119?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.from_spec" class="docs-object-method">&nbsp;</a> 
```python
from_spec(theme): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L125)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L125?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.__enter__" class="docs-object-method">&nbsp;</a> 
```python
__enter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L146)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L146?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.__exit__" class="docs-object-method">&nbsp;</a> 
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L157)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L157?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.theme" class="docs-object-method">&nbsp;</a> 
```python
@property
theme(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.add_theme" class="docs-object-method">&nbsp;</a> 
```python
add_theme(theme_name, *base_theme, **extra_styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L168)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L168?message=Update%20Docs)]
</div>

Adds a theme to the extra themes dict. At some future date we'll
        want to make it so that this does a level of validation, too.
- `theme_name`: `Any`
    >No description...
- `base_theme`: `Any`
    >No description...
- `extra_styles`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Styling.ThemeManager.resolve_theme" class="docs-object-method">&nbsp;</a> 
```python
resolve_theme(theme_name, *base_themes, **extra_styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L183)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L183?message=Update%20Docs)]
</div>

Resolves a theme so that it only uses strings for built-in styles
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Styling.ThemeManager.validate_theme" class="docs-object-method">&nbsp;</a> 
```python
validate_theme(self, theme_names, theme_styless): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L229)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L229?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.backend_themes" class="docs-object-method">&nbsp;</a> 
```python
@property
backend_themes(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.theme_names" class="docs-object-method">&nbsp;</a> 
```python
@property
theme_names(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/Styling.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots/Styling/ThemeManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots/Styling/ThemeManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots/Styling/ThemeManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots/Styling/ThemeManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Plots/Styling.py#L100?message=Update%20Docs)