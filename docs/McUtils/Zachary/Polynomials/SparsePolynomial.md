## <a id="McUtils.Zachary.Polynomials.SparsePolynomial">SparsePolynomial</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials.py#L783)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials.py#L783?message=Update%20Docs)]
</div>

A semi-symbolic representation of a polynomial of tensor
coefficients







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Zachary.Polynomials.SparsePolynomial.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, terms: dict, prefactor=1): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L788)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L788?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.scaling" class="docs-object-method">&nbsp;</a> 
```python
@property
scaling(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L793)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L793?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.expand" class="docs-object-method">&nbsp;</a> 
```python
expand(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L800)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L800?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.monomial" class="docs-object-method">&nbsp;</a> 
```python
monomial(idx, value=1): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L805)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L805?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L808)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L808?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L811)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L811?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.__add__" class="docs-object-method">&nbsp;</a> 
```python
__add__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L832)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L832?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L874)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L874?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.as_dense" class="docs-object-method">&nbsp;</a> 
```python
as_dense(self) -> McUtils.Zachary.Polynomials.DensePolynomial: 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L884)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L884?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.SparsePolynomial.shift" class="docs-object-method">&nbsp;</a> 
```python
shift(self, shift) -> McUtils.Zachary.Polynomials.DensePolynomial: 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/SparsePolynomial.py#L894)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/SparsePolynomial.py#L894?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/Polynomials/SparsePolynomial.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/Polynomials/SparsePolynomial.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/Polynomials/SparsePolynomial.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/Polynomials/SparsePolynomial.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials.py#L783?message=Update%20Docs)   
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