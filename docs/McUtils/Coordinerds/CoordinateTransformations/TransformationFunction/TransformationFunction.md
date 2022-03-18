## <a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction">TransformationFunction</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L13)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L13?message=Update%20Docs)]
</div>

The TransformationFunction class is an abstract class
It provides the scaffolding for representing a single transformation operation

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L19)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L19?message=Update%20Docs)]
</div>

Initializes a transformation function based on the transfdata
- `transfdata`: `Any`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.inverse" class="docs-object-method">&nbsp;</a> 
```python
@property
inverse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L?message=Update%20Docs)]
</div>

Returns the inverse of the transformation
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.merge" class="docs-object-method">&nbsp;</a> 
```python
merge(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L36)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L36?message=Update%20Docs)]
</div>

Tries to merge with another TransformationFunction
- `other`: `TransformationFunction`
    >a TransformationFunction to try to merge with
- `:returns`: `TransformationFunction`
    >tfunc

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.operate" class="docs-object-method">&nbsp;</a> 
```python
operate(self, coords, shift=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L47)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L47?message=Update%20Docs)]
</div>

Operates on the coords. *Must* be able to deal with a list of coordinates, optimally in an efficient manner
- `coords`: `np.ndarry`
    >the list of coordinates to apply the transformation to

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/CoordinateTransformations/TransformationFunction.py#L13?message=Update%20Docs)