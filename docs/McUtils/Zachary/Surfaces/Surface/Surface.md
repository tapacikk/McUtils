## <a id="McUtils.Zachary.Surfaces.Surface.Surface">Surface</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/Surface.py#L11)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/Surface.py#L11?message=Update%20Docs)]
</div>

This actually isn't a concrete implementation of BaseSurface.
Instead it's a class that _dispatches_ to an implementation of BaseSurface to do its core evaluations (plus it does shape checking)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.Surfaces.Surface.Surface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, data, dimension=None, base=None, **metadata): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/Surface.py#L16)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/Surface.py#L16?message=Update%20Docs)]
</div>


- `data`: `Any`
    >No description...
- `dimension`: `Any`
    >No description...
- `base`: `None | Type[BaseSurface]`
    >No description...
- `metadata`: `Any`
    >No description...

<a id="McUtils.Zachary.Surfaces.Surface.Surface.data" class="docs-object-method">&nbsp;</a> 
```python
@property
data(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/Surface.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/Surface.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Surfaces.Surface.Surface.minimize" class="docs-object-method">&nbsp;</a> 
```python
minimize(self, initial_guess=None, function_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/Surface.py#L44)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/Surface.py#L44?message=Update%20Docs)]
</div>

Provides a uniform interface for minimization, basically just dispatching to the BaseSurface implementation if provided
- `initial_guess`: `np.ndarray | None`
    >initial starting point for the minimization
- `function_options`: `None | dict`
    >No description...
- `opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Surfaces.Surface.Surface.detect_base" class="docs-object-method">&nbsp;</a> 
```python
detect_base(data, opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/Surface.py#L64)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/Surface.py#L64?message=Update%20Docs)]
</div>

Infers what type of base surface works for the data that's passed in.
        It's _super_ roughly done so...yeah generally better to pass the base class you want explicitly.
        But in the absence of that we can do this ?_?

        Basic strategy:
            1. look for options that go with specific methods
            2. look at data structures to guess
                i.   gradient as the first data arg + all data args are ndarrays -> Taylor Series
                ii.  callables as second arg -> Linear expansion or Linear fit
                iii. just like...one big array -> Interpolatin
- `data`: `tuple`
    >No description...
- `opts`: `dict`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Surfaces.Surface.Surface.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, gridpoints, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/Surface.py#L108)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/Surface.py#L108?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Surfaces/Surface/Surface.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Surfaces/Surface/Surface.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Surfaces/Surface/Surface.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/Surface/Surface.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/Surface.py#L11?message=Update%20Docs)