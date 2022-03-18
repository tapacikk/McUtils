## <a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform">AffineTransform</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L17)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L17?message=Update%20Docs)]
</div>

A simple AffineTranform implementation of the TransformationFunction abstract base class

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, tmat, shift=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L22)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L22?message=Update%20Docs)]
</div>

tmat must be a transformation matrix to work properly
- `shift`: `np.ndarray | None`
    >the shift for the transformation
- `tmat`: `np.ndarray`
    >the matrix for the linear transformation

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.transform" class="docs-object-method">&nbsp;</a> 
```python
@property
transform(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.inverse" class="docs-object-method">&nbsp;</a> 
```python
@property
inverse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L?message=Update%20Docs)]
</div>

Returns the inverse of the transformation
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.shift" class="docs-object-method">&nbsp;</a> 
```python
@property
shift(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.merge" class="docs-object-method">&nbsp;</a> 
```python
merge(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L57)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L57?message=Update%20Docs)]
</div>


- `other`: `np.ndarray or AffineTransform`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.reverse" class="docs-object-method">&nbsp;</a> 
```python
reverse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L73)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L73?message=Update%20Docs)]
</div>

Inverts the matrix
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.operate" class="docs-object-method">&nbsp;</a> 
```python
operate(self, coords, shift=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L84)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L84?message=Update%20Docs)]
</div>


- `coords`: `np.ndarry`
    >the array of coordinates passed in

<a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L119)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L119?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/AffineTransform.py#L17?message=Update%20Docs)