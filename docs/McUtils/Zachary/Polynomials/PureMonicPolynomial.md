## <a id="McUtils.Zachary.Polynomials.PureMonicPolynomial">PureMonicPolynomial</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials.py#L897)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials.py#L897?message=Update%20Docs)]
</div>









<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, terms: dict, prefactor=1, canonicalize=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L898)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L898?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.shape" class="docs-object-method">&nbsp;</a> 
```python
@property
shape(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L903)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L903?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.as_dense" class="docs-object-method">&nbsp;</a> 
```python
as_dense(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L906)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L906?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.shift" class="docs-object-method">&nbsp;</a> 
```python
shift(self, shift) -> McUtils.Zachary.Polynomials.DensePolynomial: 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L908)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L908?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.monomial" class="docs-object-method">&nbsp;</a> 
```python
monomial(idx, value=1): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L911)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L911?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.key_hash" class="docs-object-method">&nbsp;</a> 
```python
key_hash(monomial_tuple): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L915)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L915?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.canonical_key" class="docs-object-method">&nbsp;</a> 
```python
canonical_key(monomial_tuple): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L921)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L921?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.direct_multiproduct" class="docs-object-method">&nbsp;</a> 
```python
direct_multiproduct(self, other, key_value_generator): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L926)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L926?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.direct_product" class="docs-object-method">&nbsp;</a> 
```python
direct_product(self, other, key_func=None, mul=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L951)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L951?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.Polynomials.PureMonicPolynomial.__mul__" class="docs-object-method">&nbsp;</a> 
```python
__mul__(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/Polynomials/PureMonicPolynomial.py#L985)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials/PureMonicPolynomial.py#L985?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Zachary/Polynomials/PureMonicPolynomial.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Zachary/Polynomials/PureMonicPolynomial.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Zachary/Polynomials/PureMonicPolynomial.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Zachary/Polynomials/PureMonicPolynomial.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/Polynomials.py#L897?message=Update%20Docs)   
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