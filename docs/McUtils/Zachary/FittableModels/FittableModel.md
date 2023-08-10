## <a id="McUtils.Zachary.FittableModels.FittableModel">FittableModel</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L16)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L16?message=Update%20Docs)]
</div>

Defines a model that can be fit







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Zachary.FittableModels.FittableModel.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, parameters, function, pre_fit=False, covariance=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L20)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L20?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.FittableModels.FittableModel.parameters" class="docs-object-method">&nbsp;</a> 
```python
@property
parameters(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L37)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L37?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.FittableModels.FittableModel.parameter_values" class="docs-object-method">&nbsp;</a> 
```python
@property
parameter_values(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L42)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L42?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.FittableModels.FittableModel.parameter_names" class="docs-object-method">&nbsp;</a> 
```python
@property
parameter_names(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L45)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L45?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.FittableModels.FittableModel.fitted" class="docs-object-method">&nbsp;</a> 
```python
@property
fitted(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L48)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L48?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.FittableModels.FittableModel.fit" class="docs-object-method">&nbsp;</a> 
```python
fit(self, xdata, ydata=None, fitter=None, **methopts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L52)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L52?message=Update%20Docs)]
</div>
Fits the model to the data using scipy.optimize.curve_fit or a function that provides the same interface
  - `points`: `Any`
    > 
  - `methopts`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.FittableModels.FittableModel.get_parameter" class="docs-object-method">&nbsp;</a> 
```python
get_parameter(self, name): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L81)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L81?message=Update%20Docs)]
</div>
Returns the fitted value of the parameter given by 'name'
  - `name`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.FittableModels.FittableModel.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L96)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L96?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.FittableModels.FittableModel.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, xdata): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L99)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L99?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.FittableModels.FittableModel.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, xdata): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels/FittableModel.py#L103)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels/FittableModel.py#L103?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/FittableModels/FittableModel.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/FittableModels/FittableModel.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/FittableModels/FittableModel.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/FittableModels/FittableModel.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L16?message=Update%20Docs)   
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