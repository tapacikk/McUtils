## <a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface">BaseSurface</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L15)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L15?message=Update%20Docs)]
</div>

Surface base class which can be subclassed for relevant cases







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, data, dimension): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L19)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L19?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, points, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L24)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L24?message=Update%20Docs)]
</div>
Evaluates the function at the points based off of "data"
  - `points`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, gridpoints, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L35)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L35?message=Update%20Docs)]
</div>

  - `gridpoints`: `np.ndarray`
    > 
  - `kwargs`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.Surfaces.BaseSurface.BaseSurface.minimize" class="docs-object-method">&nbsp;</a> 
```python
minimize(self, initial_guess, function_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L64)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface/BaseSurface.py#L64?message=Update%20Docs)]
</div>
Just calls into `scipy.optimize.minimize` as the default implementation
  - `initial_guess`: `np.ndarray`
    > starting position for the minimzation
  - `function_options`: `dict | None`
    > 
  - `opts`: `Any`
    > 
  - `:returns`: `_`
    >
 </div>
</div>












---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/BaseSurface/BaseSurface.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L15?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>