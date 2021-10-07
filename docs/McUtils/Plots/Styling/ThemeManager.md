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

<a id="McUtils.Plots.Styling.ThemeManager.from_spec" class="docs-object-method">&nbsp;</a>
```python
from_spec(theme): 
```

<a id="McUtils.Plots.Styling.ThemeManager.__enter__" class="docs-object-method">&nbsp;</a>
```python
__enter__(self): 
```

<a id="McUtils.Plots.Styling.ThemeManager.__exit__" class="docs-object-method">&nbsp;</a>
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```

<a id="McUtils.Plots.Styling.ThemeManager.theme" class="docs-object-method">&nbsp;</a>
```python
@property
theme(self): 
```

<a id="McUtils.Plots.Styling.ThemeManager.add_theme" class="docs-object-method">&nbsp;</a>
```python
add_theme(theme_name, *base_theme, **extra_styles): 
```
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
Resolves a theme so that it only uses strings for built-in styles
- `:returns`: `_`
    >No description...

<a id="McUtils.Plots.Styling.ThemeManager.validate_theme" class="docs-object-method">&nbsp;</a>
```python
validate_theme(self, theme_names, theme_styless): 
```

<a id="McUtils.Plots.Styling.ThemeManager.backend_themes" class="docs-object-method">&nbsp;</a>
```python
@property
backend_themes(self): 
```

<a id="McUtils.Plots.Styling.ThemeManager.theme_names" class="docs-object-method">&nbsp;</a>
```python
@property
theme_names(self): 
```

### Examples


