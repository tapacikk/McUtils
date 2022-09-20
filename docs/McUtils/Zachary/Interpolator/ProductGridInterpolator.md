## <a id="McUtils.Zachary.Interpolator.ProductGridInterpolator">ProductGridInterpolator</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator.py#L55)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L55?message=Update%20Docs)]
</div>

A set of interpolators that support interpolation
on a regular (tensor product) grid







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, grids, vals, caller=None, order=None, extrapolate=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator/ProductGridInterpolator.py#L61)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator/ProductGridInterpolator.py#L61?message=Update%20Docs)]
</div>

  - `grids`: `Any`
    > 
  - `points`: `Any`
    > 
  - `caller`: `Any`
    > 
  - `order`: `int | Iterable[int]`
    >


<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.construct_ndspline" class="docs-object-method">&nbsp;</a> 
```python
construct_ndspline(grids, vals, order, extrapolate=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator/ProductGridInterpolator.py#L95)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator/ProductGridInterpolator.py#L95?message=Update%20Docs)]
</div>
Builds a tensor product ndspline by constructing a product of 1D splines
  - `grids`: `Iterable[np.ndarray]`
    > grids for each dimension independently
  - `vals`: `np.ndarray`
    > 
  - `order`: `int | Iterable[int]`
    > 
  - `:returns`: `interpolate.NdPPoly`
    >


<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator/ProductGridInterpolator.py#L137)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator/ProductGridInterpolator.py#L137?message=Update%20Docs)]
</div>

  - `args`: `Any`
    > 
  - `kwargs`: `Any`
    > 
  - `:returns`: `np.ndarray`
    >


<a id="McUtils.Zachary.Interpolator.ProductGridInterpolator.derivative" class="docs-object-method">&nbsp;</a> 
```python
derivative(self, order): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Interpolator/ProductGridInterpolator.py#L148)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator/ProductGridInterpolator.py#L148?message=Update%20Docs)]
</div>

  - `order`: `Any`
    > 
  - `:returns`: `ProductGridInterpolator`
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Interpolator/ProductGridInterpolator.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Interpolator/ProductGridInterpolator.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Interpolator/ProductGridInterpolator.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Interpolator/ProductGridInterpolator.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Interpolator.py#L55?message=Update%20Docs)   
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