## <a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction">TransformationFunction</a>
The TransformationFunction class is an abstract class
It provides the scaffolding for representing a single transformation operation

### Properties and Methods
<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L19)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L19?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L?message=Update%20Docs)]
</div>

Returns the inverse of the transformation
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.merge" class="docs-object-method">&nbsp;</a> 
```python
merge(self, other): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L36)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L36?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L47)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Coordinerds/CoordinateTransformations/TransformationFunction.py#L47?message=Update%20Docs)]
</div>

Operates on the coords. *Must* be able to deal with a list of coordinates, optimally in an efficient manner
- `coords`: `np.ndarry`
    >the list of coordinates to apply the transformation to





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction.py?message=Update%20Docs)