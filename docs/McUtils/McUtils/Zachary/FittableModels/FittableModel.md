## <a id="McUtils.McUtils.Zachary.FittableModels.FittableModel">FittableModel</a>
Defines a model that can be fit

### Properties and Methods
<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, parameters, function, pre_fit=False, covariance=None): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.parameters" class="docs-object-method">&nbsp;</a>
```python
@property
parameters(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.parameter_values" class="docs-object-method">&nbsp;</a>
```python
@property
parameter_values(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.parameter_names" class="docs-object-method">&nbsp;</a>
```python
@property
parameter_names(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.fitted" class="docs-object-method">&nbsp;</a>
```python
@property
fitted(self): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.fit" class="docs-object-method">&nbsp;</a>
```python
fit(self, xdata, ydata=None, fitter=None, **methopts): 
```
Fits the model to the data using scipy.optimize.curve_fit or a function that provides the same interface
- `points`: `Any`
    >No description...
- `methopts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.get_parameter" class="docs-object-method">&nbsp;</a>
```python
get_parameter(self, name): 
```
Returns the fitted value of the parameter given by 'name'
- `name`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.__getitem__" class="docs-object-method">&nbsp;</a>
```python
__getitem__(self, item): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.evaluate" class="docs-object-method">&nbsp;</a>
```python
evaluate(self, xdata): 
```

<a id="McUtils.McUtils.Zachary.FittableModels.FittableModel.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, xdata): 
```

### Examples


