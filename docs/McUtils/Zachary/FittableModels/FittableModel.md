## <a id="McUtils.Zachary.FittableModels.FittableModel">FittableModel</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L16)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L16?message=Update%20Docs)]
</div>

Defines a model that can be fit

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.FittableModels.FittableModel.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, parameters, function, pre_fit=False, covariance=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L20)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L20?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.FittableModel.parameters" class="docs-object-method">&nbsp;</a> 
```python
@property
parameters(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.FittableModel.parameter_values" class="docs-object-method">&nbsp;</a> 
```python
@property
parameter_values(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.FittableModel.parameter_names" class="docs-object-method">&nbsp;</a> 
```python
@property
parameter_names(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.FittableModel.fitted" class="docs-object-method">&nbsp;</a> 
```python
@property
fitted(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.FittableModel.fit" class="docs-object-method">&nbsp;</a> 
```python
fit(self, xdata, ydata=None, fitter=None, **methopts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L52)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L52?message=Update%20Docs)]
</div>

Fits the model to the data using scipy.optimize.curve_fit or a function that provides the same interface
- `points`: `Any`
    >No description...
- `methopts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.FittableModels.FittableModel.get_parameter" class="docs-object-method">&nbsp;</a> 
```python
get_parameter(self, name): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L81)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L81?message=Update%20Docs)]
</div>

Returns the fitted value of the parameter given by 'name'
- `name`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.FittableModels.FittableModel.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L96)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L96?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.FittableModel.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, xdata): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L99)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L99?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.FittableModel.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, xdata): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L103)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L103?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/FittableModels/FittableModel.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/FittableModels/FittableModel.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/FittableModels/FittableModel.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/FittableModels/FittableModel.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L16?message=Update%20Docs)