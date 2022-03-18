## <a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform">CoordinateTransform</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L25)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L25?message=Update%20Docs)]
</div>

The CoordinateTransform class provides a simple, general way to represent a
compound coordinate transformation.
In general, it's basically just a wrapper chaining together a number of TransformationFunctions.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *transforms): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L32)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L32?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.is_affine" class="docs-object-method">&nbsp;</a> 
```python
@property
is_affine(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.transformation_function" class="docs-object-method">&nbsp;</a> 
```python
@property
transformation_function(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L?message=Update%20Docs)]
</div>


- `:returns`: `TransformationFunction`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.transforms" class="docs-object-method">&nbsp;</a> 
```python
@property
transforms(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.apply" class="docs-object-method">&nbsp;</a> 
```python
apply(self, coords, shift=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L54)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L54?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self, coords, shift=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L57)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L57?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.condense_transforms" class="docs-object-method">&nbsp;</a> 
```python
condense_transforms(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L63)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L63?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.inverse" class="docs-object-method">&nbsp;</a> 
```python
@property
inverse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.parse_transform" class="docs-object-method">&nbsp;</a> 
```python
parse_transform(tf): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L72)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L72?message=Update%20Docs)]
</div>

Provides a way to "tag" a transformation
- `tf`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateTransformations/CoordinateTransform/CoordinateTransform.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateTransformations/CoordinateTransform/CoordinateTransform.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateTransformations/CoordinateTransform/CoordinateTransform.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateTransformations/CoordinateTransform/CoordinateTransform.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/CoordinateTransform.py#L25?message=Update%20Docs)