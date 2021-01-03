## <a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform">CoordinateTransform</a>
The CoordinateTransform class provides a simple, general way to represent a
compound coordinate transformation.
In general, it's basically just a wrapper chaining together a number of TransformationFunctions.

### Properties and Methods
<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, *transforms): 
```

<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.transformation_function" class="docs-object-method">&nbsp;</a>
```python
@property
transformation_function(self): 
```

- `:returns`: `TransformationFunction`
    >No description...

<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.transforms" class="docs-object-method">&nbsp;</a>
```python
@property
transforms(self): 
```

<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, coords, shift=True): 
```

<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self, coords, shift=True): 
```

<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.condense_transforms" class="docs-object-method">&nbsp;</a>
```python
condense_transforms(self): 
```

<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.inverse" class="docs-object-method">&nbsp;</a>
```python
@property
inverse(self): 
```

<a id="McUtils.McUtils.Coordinerds.CoordinateTransformations.CoordinateTransform.CoordinateTransform.parse_transform" class="docs-object-method">&nbsp;</a>
```python
parse_transform(tf): 
```
Provides a way to "tag" a transformation
- `tf`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples


