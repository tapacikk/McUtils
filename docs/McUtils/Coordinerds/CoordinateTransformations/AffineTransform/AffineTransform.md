## <a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform">AffineTransform</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform.py#L17)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform.py#L17?message=Update%20Docs)]
</div>

A simple AffineTranform implementation of the TransformationFunction abstract base class







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, tmat, shift=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L22)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L22?message=Update%20Docs)]
</div>
tmat must be a transformation matrix to work properly
  - `shift`: `np.ndarray | None`
    > the shift for the transformation
  - `tmat`: `np.ndarray`
    > the matrix for the linear transformation


<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.transform" class="docs-object-method">&nbsp;</a> 
```python
@property
transform(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L34)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L34?message=Update%20Docs)]
</div>


<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.inverse" class="docs-object-method">&nbsp;</a> 
```python
@property
inverse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L38)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L38?message=Update%20Docs)]
</div>
Returns the inverse of the transformation
  - `:returns`: `_`
    >


<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.shift" class="docs-object-method">&nbsp;</a> 
```python
@property
shift(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L47)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L47?message=Update%20Docs)]
</div>


<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.merge" class="docs-object-method">&nbsp;</a> 
```python
merge(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L57)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L57?message=Update%20Docs)]
</div>

  - `other`: `np.ndarray or AffineTransform`
    >


<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.reverse" class="docs-object-method">&nbsp;</a> 
```python
reverse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L73)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L73?message=Update%20Docs)]
</div>
Inverts the matrix
  - `:returns`: `_`
    >


<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.operate" class="docs-object-method">&nbsp;</a> 
```python
operate(self, coords, shift=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L84)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L84?message=Update%20Docs)]
</div>

  - `coords`: `np.ndarry`
    > the array of coordinates passed in


<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L119)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.py#L119?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/AffineTransform.py#L17?message=Update%20Docs)   
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