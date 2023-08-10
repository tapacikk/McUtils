## <a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface">LinearFitSurface</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface.py#L155)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L155?message=Update%20Docs)]
</div>

A surface built off of a LinearExpansionSurface, but done by fitting.
The basis selection







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, points, basis=None, order=4, dimension=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface/LinearFitSurface.py#L160)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface/LinearFitSurface.py#L160?message=Update%20Docs)]
</div>

  - `points`: `np.ndarray`
    > a set of points to fit to
  - `basis`: `Iterable[function] | None`
    > a basis of functions to use (defaults to power series)


<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, points, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface/LinearFitSurface.py#L182)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface/LinearFitSurface.py#L182?message=Update%20Docs)]
</div>

  - `points`: `Any`
    > 
  - `kwargs`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.Surfaces.BaseSurface.LinearFitSurface.minimize" class="docs-object-method">&nbsp;</a> 
```python
minimize(self, initial_guess=None, function_options=None, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Surfaces/BaseSurface/LinearFitSurface.py#L196)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface/LinearFitSurface.py#L196?message=Update%20Docs)]
</div>
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/Surfaces/BaseSurface/LinearFitSurface.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Surfaces/BaseSurface.py#L155?message=Update%20Docs)   
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