## <a id="McUtils.Zachary.Surfaces.BaseSurface.InterpolatedSurface">InterpolatedSurface</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L201)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L201?message=Update%20Docs)]
</div>

A surface that operates by doing an interpolation of passed mesh data

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.Surfaces.BaseSurface.InterpolatedSurface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, xdata, ydata=None, dimension=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L205)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L205?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.Surfaces.BaseSurface.InterpolatedSurface.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, points, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L215)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L215?message=Update%20Docs)]
</div>

We delegate all the dirty work to the Interpolator so hopefully that's working...
- `points`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Surfaces.BaseSurface.InterpolatedSurface.minimize" class="docs-object-method">&nbsp;</a> 
```python
minimize(self, initial_guess=None, function_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/Surfaces/BaseSurface.py#L227)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L227?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Surfaces/BaseSurface/InterpolatedSurface.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Surfaces/BaseSurface/InterpolatedSurface.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Surfaces/BaseSurface/InterpolatedSurface.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/BaseSurface/InterpolatedSurface.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/Surfaces/BaseSurface.py#L201?message=Update%20Docs)