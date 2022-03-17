## <a id="McUtils.Plots.Styling.ThemeManager">ThemeManager</a>
Simple manager class for plugging into themes in a semi-background agnostic way

### Properties and Methods
```python
extra_themes: dict
```
<a id="McUtils.Plots.Styling.ThemeManager.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *theme_names, backend=<Backends.MPL: 'matplotlib'>, graphics_styles=None, **extra_styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L41)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L41?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.from_spec" class="docs-object-method">&nbsp;</a> 
```python
from_spec(theme): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L47)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L47?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.__enter__" class="docs-object-method">&nbsp;</a> 
```python
__enter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L68)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L68?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.__exit__" class="docs-object-method">&nbsp;</a> 
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L79)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L79?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.theme" class="docs-object-method">&nbsp;</a> 
```python
@property
theme(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.add_theme" class="docs-object-method">&nbsp;</a> 
```python
add_theme(theme_name, *base_theme, **extra_styles): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L90)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L90?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L105)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L105?message=Update%20Docs)]
</div>

Resolves a theme so that it only uses strings for built-in styles
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Styling.ThemeManager.validate_theme" class="docs-object-method">&nbsp;</a> 
```python
validate_theme(self, theme_names, theme_styless): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L151)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L151?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.backend_themes" class="docs-object-method">&nbsp;</a> 
```python
@property
backend_themes(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Plots.Styling.ThemeManager.theme_names" class="docs-object-method">&nbsp;</a> 
```python
@property
theme_names(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Plots/Styling.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Plots/Styling.py#L?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Plots/Styling/ThemeManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Plots/Styling/ThemeManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Plots/Styling/ThemeManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Plots/Styling/ThemeManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Plots/Styling.py?message=Update%20Docs)