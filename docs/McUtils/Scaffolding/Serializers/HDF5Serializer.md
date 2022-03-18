## <a id="McUtils.Scaffolding.Serializers.HDF5Serializer">HDF5Serializer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L650)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L650?message=Update%20Docs)]
</div>

Defines a serializer that can prep/dump python data to HDF5.
To minimize complexity, we always use NumPy & Pseudopickle as an interface layer.
This restricts what we can serialize, but generally in insignificant ways.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_extension: str
```
<a id="McUtils.Scaffolding.Serializers.HDF5Serializer.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, allow_pickle=True, psuedopickler=None, converters=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L657)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L657?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.HDF5Serializer.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L672)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L672?message=Update%20Docs)]
</div>

Converts data into format that can be serialized easily
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.HDF5Serializer.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, file, data, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L807)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L807?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.HDF5Serializer.deconvert" class="docs-object-method">&nbsp;</a> 
```python
deconvert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L819)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L819?message=Update%20Docs)]
</div>

Converts an HDF5 Dataset into a NumPy array or Group into a dict
- `data`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.HDF5Serializer.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(self, file, key=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L844)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L844?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [HDF5Serialization](#HDF5Serialization)
- [HDF5PseudoPickleSerialization](#HDF5PseudoPickleSerialization)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
#### <a class="collapse-link" data-toggle="collapse" href="#test-setup">Setup</a> <a class="float-right" data-toggle="collapse" href="#test-setup"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="test-setup" markdown="1">

Before we can run our examples we should get a bit of setup out of the way.
Since these examples were harvested from the unit tests not all pieces
will be necessary for all situations.
```python
from Peeves.TestUtils import *
from McUtils.Scaffolding import *
import McUtils.Parsers as parsers
from unittest import TestCase
import numpy as np, io, os, sys, tempfile as tmpf
```

All tests are wrapped in a test class
```python
class ScaffoldingTests(TestCase):
    class DataHolderClass:
        def __init__(self, **keys):
            self.data = keys
        def to_state(self, serializer=None):
            return self.data
        @classmethod
        def from_state(cls, state, serializer=None):
            return cls(**state)
```

 </div>
</div>

#### <a name="HDF5Serialization">HDF5Serialization</a>
```python
    def test_HDF5Serialization(self):
        tmp = io.BytesIO()
        serializer = HDF5Serializer()

        data = [1, 2, 3]
        serializer.serialize(tmp, data)
        loaded = serializer.deserialize(tmp)
        self.assertEquals(loaded.tolist(), data)

        serializer.serialize(tmp, {
            "blebby": {
                "frebby": {
                    "clebby":data
                }
            }
        })
        loaded = serializer.deserialize(tmp, key='blebby')
        self.assertEquals(loaded['frebby']['clebby'].tolist(), data)

        mixed_data = [
            [1, 2, 3],
            "garbage",
            {"temps":[1., 2., 3.]}
            ]
        serializer.serialize(tmp, dict(mixed_data=mixed_data))

        loaded = serializer.deserialize(tmp, key='mixed_data')
        self.assertEquals(mixed_data, [
            loaded[0].tolist(),
            loaded[1].tolist().decode('utf-8'),
            {k:v.tolist() for k,v in loaded[2].items()}
        ])
```
#### <a name="HDF5PseudoPickleSerialization">HDF5PseudoPickleSerialization</a>
```python
    def test_HDF5PseudoPickleSerialization(self):

        from McUtils.Numputils import SparseArray

        tmp = io.BytesIO()
        serializer = HDF5Serializer()

        data = SparseArray.from_diag([1, 2, 3, 4])

        serializer.serialize(tmp, data)
        tmp.seek(0)
        loaded = serializer.deserialize(tmp)

        self.assertTrue(np.allclose(loaded.asarray(), data.asarray()))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Serializers/HDF5Serializer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Serializers/HDF5Serializer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Serializers/HDF5Serializer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/HDF5Serializer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L650?message=Update%20Docs)