## <a id="McUtils.Coordinerds.CoordinateTransformations.AffineTransform.AffineTransform">AffineTransform</a>
A simple AffineTranform implementation of the TransformationFunction abstract base class

### Properties and Methods
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





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Coordinerds/CoordinateTransformations/AffineTransform.py?message=Update%20Docs)