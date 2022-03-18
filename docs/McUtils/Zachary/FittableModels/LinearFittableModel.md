## <a id="McUtils.Zachary.FittableModels.LinearFittableModel">LinearFittableModel</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L106)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L106?message=Update%20Docs)]
</div>

Defines a class of models that can be expressed as linear expansions of basis functions.
We _could_ define an alternate fit function by explicitly building & fitting a design matrix, but I think we're good on that for now

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Zachary.FittableModels.LinearFittableModel.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, basis, initial_params=None, pre_fit=False, covariance=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L112)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L112?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.LinearFittableModel.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, xdata): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/FittableModels.py#L124)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L124?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/FittableModels/LinearFittableModel.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/FittableModels/LinearFittableModel.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/FittableModels/LinearFittableModel.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/FittableModels/LinearFittableModel.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/FittableModels.py#L106?message=Update%20Docs)