## <a id="McUtils.Zachary.Polynomials.DensePolynomial">DensePolynomial</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials.py#L53)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials.py#L53?message=Update%20Docs)]
</div>

A straightforward dense n-dimensional polynomial data structure with
multiplications and shifts







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Zachary.Polynomials.DensePolynomial.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, coeffs, prefactor=None, shift=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L58)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L58?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L68)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L68?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.from_tensors" class="docs-object-method">&nbsp;</a> 
```python
from_tensors(tensors, prefactor=None, shift=None, rescale=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L70)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L70?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L77)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L77?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.scaling" class="docs-object-method">&nbsp;</a> 
```python
@property
scaling(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L81)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L81?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.coeffs" class="docs-object-method">&nbsp;</a> 
```python
@property
coeffs(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L88)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L88?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other) -> 'DensePolynomial': 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L101)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L101?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.__add__" class="docs-object-method">&nbsp;</a> 
```python
__add__(self, other) -> 'DensePolynomial': 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L125)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L125?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.shift" class="docs-object-method">&nbsp;</a> 
```python
shift(self, shift) -> 'DensePolynomial': 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L160)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L160?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.extract_tensors" class="docs-object-method">&nbsp;</a> 
```python
extract_tensors(coeffs, rescale=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L211)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L211?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.condense_tensors" class="docs-object-method">&nbsp;</a> 
```python
condense_tensors(tensors, rescale=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L247)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L247?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.coefficient_tensors" class="docs-object-method">&nbsp;</a> 
```python
@property
coefficient_tensors(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L277)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L277?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.unscaled_coefficient_tensors" class="docs-object-method">&nbsp;</a> 
```python
@property
unscaled_coefficient_tensors(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L282)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L282?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.transform" class="docs-object-method">&nbsp;</a> 
```python
transform(self, lin_transf): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L287)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L287?message=Update%20Docs)]
</div>
Applies (for now) a linear transformation to the polynomial


<a id="McUtils.Zachary.Polynomials.DensePolynomial.outer" class="docs-object-method">&nbsp;</a> 
```python
outer(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L300)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L300?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.deriv" class="docs-object-method">&nbsp;</a> 
```python
deriv(self, coord): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L306)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L306?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.DensePolynomial.clip" class="docs-object-method">&nbsp;</a> 
```python
clip(self, threshold=1e-15): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/DensePolynomial.py#L321)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/DensePolynomial.py#L321?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/Polynomials/DensePolynomial.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/Polynomials/DensePolynomial.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/Polynomials/DensePolynomial.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/Polynomials/DensePolynomial.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials.py#L53?message=Update%20Docs)   
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