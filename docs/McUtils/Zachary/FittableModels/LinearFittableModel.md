## <a id="McUtils.Zachary.FittableModels.LinearFittableModel">LinearFittableModel</a>
Defines a class of models that can be expressed as linear expansions of basis functions.
We _could_ define an alternate fit function by explicitly building & fitting a design matrix, but I think we're good on that for now

### Properties and Methods
<a id="McUtils.Zachary.FittableModels.LinearFittableModel.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, basis, initial_params=None, pre_fit=False, covariance=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/FittableModels.py#L112)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/FittableModels.py#L112?message=Update%20Docs)]
</div>

<a id="McUtils.Zachary.FittableModels.LinearFittableModel.evaluate" class="docs-object-method">&nbsp;</a> 
```python
evaluate(self, xdata): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Zachary/FittableModels.py#L124)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Zachary/FittableModels.py#L124?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Zachary/FittableModels/LinearFittableModel.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Zachary/FittableModels/LinearFittableModel.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Zachary/FittableModels/LinearFittableModel.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Zachary/FittableModels/LinearFittableModel.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Zachary/FittableModels.py?message=Update%20Docs)