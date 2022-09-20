## <a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion">FunctionExpansion</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions.py#L18)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions.py#L18?message=Update%20Docs)]
</div>

A class for handling expansions of an arbitrary function
given a set of derivatives and also allows for coordinate
transformations if the appropriate derivatives are supplied.

Can be used to handle multiple expansions simultaneously, but requires
that all expansions are provided up to the same order.







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse " id="methods" markdown="1">
 ```python
CoordinateTransforms: CoordinateTransforms
FunctionDerivatives: FunctionDerivatives
```
<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, derivatives, transforms=None, center=None, ref=0, weight_coefficients=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L29)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L29?message=Update%20Docs)]
</div>

  - `derivatives`: `Iterable[np.ndarray | Tensor]`
    > Derivatives of the function being expanded
  - `transforms`: `Iterable[np.ndarray | Tensor] | None`
    > Jacobian and higher order derivatives in the coordinates
  - `center`: `np.ndarray | None`
    > the reference point for expanding aobut
  - `ref`: `float | np.ndarray`
    > the reference point value for shifting the expansion
  - `weight_coefficients`: `bool`
    > whether the derivative terms need to be weighted or not


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.center" class="docs-object-method">&nbsp;</a> 
```python
@property
center(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L59)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L59?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.is_multiexpansion" class="docs-object-method">&nbsp;</a> 
```python
@property
is_multiexpansion(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L62)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L62?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.multiexpansion" class="docs-object-method">&nbsp;</a> 
```python
multiexpansion(*expansions): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L65)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L65?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand_function" class="docs-object-method">&nbsp;</a> 
```python
expand_function(f, point, order=4, basis=None, function_shape=None, transforms=None, weight_coefficients=True, **fd_options): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L73)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L73?message=Update%20Docs)]
</div>
Expands a function about a point up to the given order
  - `f`: `function`
    > 
  - `point`: `np.ndarray | CoordinateSet`
    > 
  - `order`: `int`
    > 
  - `basis`: `None | CoordinateSystem`
    > 
  - `fd_options`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expansion_tensors" class="docs-object-method">&nbsp;</a> 
```python
@property
expansion_tensors(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L110)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L110?message=Update%20Docs)]
</div>
Provides the tensors that will contracted
  - `:returns`: `Iterable[np.ndarray]`
    >


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_expansions" class="docs-object-method">&nbsp;</a> 
```python
get_expansions(self, coords, squeeze=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L125)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L125?message=Update%20Docs)]
</div>

  - `coords`: `np.ndarray | CoordinateSet`
    > Coordinates to evaluate the expansion at
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand" class="docs-object-method">&nbsp;</a> 
```python
expand(self, coords, squeeze=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L174)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L174?message=Update%20Docs)]
</div>
Returns a numerical value for the expanded coordinates
  - `coords`: `np.ndarray`
    > 
  - `:returns`: `float | np.ndarray`
    >


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, coords, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L186)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L186?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.deriv" class="docs-object-method">&nbsp;</a> 
```python
deriv(self, which=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L246)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions/FunctionExpansion.py#L246?message=Update%20Docs)]
</div>
Computes the derivative(s) of the expansion(s) with respect to the
supplied coordinates (`which=None` means compute the gradient)
  - `which`: `Any`
    > 
  - `:returns`: `_`
    >
 </div>
</div>




## Examples
## <a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion">FunctionExpansion</a>
A class for handling expansions of an internal coordinate potential up to 4th order
Uses Cartesian derivative matrices and the Cartesian <-> Internal normal mode Jacobian

### Properties and Methods
```python
expand_function: method
CoordinateTransforms: type
FunctionDerivatives: type
```
<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, derivatives, transforms=None, center=None, ref=0, weight_coefficients=True): 
```

- `derivatives`: `Iterable[np.ndarray | Tensor]`
    >Derivatives of the function being expanded
- `transforms`: `Iterable[np.ndarray | Tensor] | None`
    >Jacobian and higher order derivatives in the coordinates
- `center`: `np.ndarray | None`
    >the reference point for expanding aobut
- `ref`: `float | np.ndarray`
    >the reference point value for shifting the expansion
- `weight_coefficients`: `bool`
    >whether the derivative terms need to be weighted or not

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.tensors" class="docs-object-method">&nbsp;</a>
```python
@property
tensors(self): 
```

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_expansions" class="docs-object-method">&nbsp;</a>
```python
get_expansions(self, coords, squeeze=True): 
```

- `coords`: `np.ndarray | CoordinateSet`
    >Coordinates to evaluate the expansion at
- `:returns`: `_`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.expand" class="docs-object-method">&nbsp;</a>
```python
expand(self, coords, squeeze=True): 
```
Returns a numerical value for the expanded coordinates
- `coords`: `np.ndarray`
    >No description...
- `:returns`: `float | np.ndarray`
    >No description...

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, coords, **kw): 
```

<a id="McUtils.Zachary.Taylor.FunctionExpansions.FunctionExpansion.get_tensor" class="docs-object-method">&nbsp;</a>
```python
get_tensor(self, i): 
```
Defines the overall tensors of derivatives
- `i`: `order of derivative tensor to provide`
    >No description...
- `:returns`: `Tensor`
    >No description...

### Examples


___

[Edit Examples](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Examples](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Template](https://github.com/McCoyGroup/References/edit/gh-pages/Documentation/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) or 
[Create New Template](https://github.com/McCoyGroup/References/new/gh-pages/?filename=Documentation/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions.py?message=Update%20Docs)






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
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/Taylor/FunctionExpansions/FunctionExpansion.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Taylor/FunctionExpansions.py#L18?message=Update%20Docs)   
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