## <a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpression">TensorExpression</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter.py#L11)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter.py#L11?message=Update%20Docs)]
</div>









<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 ```python
Term: Term
SumTerm: SumTerm
ScalingTerm: ScalingTerm
PowerTerm: PowerTerm
FlippedTerm: FlippedTerm
AxisShiftTerm: AxisShiftTerm
ContractionTerm: ContractionTerm
InverseTerm: InverseTerm
TraceTerm: TraceTerm
DeterminantTerm: DeterminantTerm
VectorNormTerm: VectorNormTerm
CoordinateVectorTerm: CoordinateVectorTerm
ConstantMatrixTerm: ConstantMatrixTerm
ScalarFunctionTerm: ScalarFunctionTerm
PolynomialTerm: PolynomialTerm
```
<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpression.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, expr: 'TensorExpression.Term', vars: dict = None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L12)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L12?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpression.eval" class="docs-object-method">&nbsp;</a> 
```python
eval(self, subs: dict = None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L15)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L15?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpression.primitives" class="docs-object-method">&nbsp;</a> 
```python
@property
primitives(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L33)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L33?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpression.walk" class="docs-object-method">&nbsp;</a> 
```python
walk(self, callback): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L36)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L36?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpression.get_prims" class="docs-object-method">&nbsp;</a> 
```python
get_prims(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L44)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L44?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.TensorDerivativeConverter.TensorExpression.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L51)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter/TensorExpression.py#L51?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/TensorDerivativeConverter/TensorExpression.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/TensorDerivativeConverter/TensorExpression.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/TensorDerivativeConverter/TensorExpression.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/TensorDerivativeConverter/TensorExpression.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/TensorDerivativeConverter.py#L11?message=Update%20Docs)   
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