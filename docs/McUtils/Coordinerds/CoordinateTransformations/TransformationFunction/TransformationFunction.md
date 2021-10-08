## <a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction">TransformationFunction</a>
The TransformationFunction class is an abstract class
It provides the scaffolding for representing a single transformation operation

### Properties and Methods
<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self): 
```
Initializes a transformation function based on the transfdata
- `transfdata`: `Any`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.inverse" class="docs-object-method">&nbsp;</a>
```python
@property
inverse(self): 
```
Returns the inverse of the transformation
- `:returns`: `_`
    >No description...

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.merge" class="docs-object-method">&nbsp;</a>
```python
merge(self, other): 
```
Tries to merge with another TransformationFunction
- `other`: `TransformationFunction`
    >a TransformationFunction to try to merge with
- `:returns`: `TransformationFunction`
    >tfunc

<a id="McUtils.Coordinerds.CoordinateTransformations.TransformationFunction.TransformationFunction.operate" class="docs-object-method">&nbsp;</a>
```python
operate(self, coords, shift=True): 
```
Operates on the coords. *Must* be able to deal with a list of coordinates, optimally in an efficient manner
- `coords`: `np.ndarry`
    >the list of coordinates to apply the transformation to





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Coordinerds/CoordinateTransformations/TransformationFunction.py?message=Update%20Docs)