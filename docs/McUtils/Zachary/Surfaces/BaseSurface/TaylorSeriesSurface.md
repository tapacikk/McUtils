## <a id="McUtils.Zachary.Surfaces.BaseSurface.TaylorSeriesSurface">TaylorSeriesSurface</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L91)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L91?message=Update%20Docs)]
</div>

A surface with an evaluator built off of a Taylor series expansion

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.Surfaces.BaseSurface.TaylorSeriesSurface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *derivs, dimension=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L95)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L95?message=Update%20Docs)]
</div>


- `data`: `Any`
    >derivs or a tuple of derivs + options
- `dimension`: `Any`
    >No description...

<a id="McUtils.Zachary.Surfaces.BaseSurface.TaylorSeriesSurface.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, points, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L110)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L110?message=Update%20Docs)]
</div>

Since the Taylor expansion stuff is already built out this is super easy
- `points`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Surfaces/BaseSurface/TaylorSeriesSurface.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Surfaces/BaseSurface/TaylorSeriesSurface.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Surfaces/BaseSurface/TaylorSeriesSurface.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/BaseSurface/TaylorSeriesSurface.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L91?message=Update%20Docs)