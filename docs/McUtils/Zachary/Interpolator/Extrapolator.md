## <a id="McUtils.Zachary.Interpolator.Extrapolator">Extrapolator</a>
A general purpose that takes your data and just extrapolates it.
This currently only exists in template format.

### Properties and Methods
<a id="McUtils.Zachary.Interpolator.Extrapolator.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, extrapolation_function, warning=False, **opts): 
```

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

<a id="McUtils.Zachary.Interpolator.Extrapolator.find_extrapolated_points" class="docs-object-method">&nbsp;</a>
```python
find_extrapolated_points(self, gps, vals, extrap_value=nan): 
```
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

<a id="McUtils.Zachary.Interpolator.Extrapolator.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, *args, **kwargs): 
```

### Examples


