## <a id="McUtils.Zachary.FittableModels.LinearFitBasis">LinearFitBasis</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L129)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L129?message=Update%20Docs)]
</div>

Provides a container to build bases of functions for fitting.
Asks for a generator for each dimension, which is just a function that takes an integer and returns a basis function at that order.
Product functions are taken up to some max order

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *generators, order=3): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L135)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L135?message=Update%20Docs)]
</div>


- `generators`: `Iterable[function]`
    >the generating functions for the bases in each dimenion
- `order`: `int`
    >the maximum order for the basis functions (currently turning off coupling isn't possible, but that could come)

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.functions" class="docs-object-method">&nbsp;</a> 
```python
@property
functions(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.names" class="docs-object-method">&nbsp;</a> 
```python
@property
names(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.order" class="docs-object-method">&nbsp;</a> 
```python
@property
order(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.construct_basis" class="docs-object-method">&nbsp;</a> 
```python
construct_basis(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L171)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L171?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.<lambda>" class="docs-object-method">&nbsp;</a> 
```python
fourier_series(x, k): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L192)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L192?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.LinearFitBasis.<lambda>" class="docs-object-method">&nbsp;</a> 
```python
power_series(x, n): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L193)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L193?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/FittableModels/LinearFitBasis.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/FittableModels/LinearFitBasis.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/FittableModels/LinearFitBasis.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/FittableModels/LinearFitBasis.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L129?message=Update%20Docs)