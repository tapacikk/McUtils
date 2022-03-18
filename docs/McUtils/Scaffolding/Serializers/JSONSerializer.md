## <a id="McUtils.Scaffolding.Serializers.JSONSerializer">JSONSerializer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L296)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L296?message=Update%20Docs)]
</div>

A serializer that makes dumping data to JSON simpler

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_extension: str
BaseEncoder: type
```
<a id="McUtils.Scaffolding.Serializers.JSONSerializer.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, encoder=None, allow_pickle=True, pseudopickler=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L326)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L326?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.JSONSerializer.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L333)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L333?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.JSONSerializer.deconvert" class="docs-object-method">&nbsp;</a> 
```python
deconvert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L335)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L335?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.JSONSerializer.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, file, data, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L337)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L337?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.JSONSerializer.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(self, file, key=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Serializers.py#L341)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L341?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [JSONSerialization](#JSONSerialization)
- [JSONPseudoPickleSerialization](#JSONPseudoPickleSerialization)

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

#### <a name="JSONSerialization">JSONSerialization</a>
```python
    def test_JSONSerialization(self):
        tmp = io.StringIO()
        serializer = JSONSerializer()

        data = [1, 2, 3]
        serializer.serialize(tmp, data)
        tmp.seek(0)
        loaded = serializer.deserialize(tmp)
        self.assertEquals(loaded, data)

        tmp = io.StringIO()
        serializer.serialize(tmp, {
            "blebby": {
                "frebby": {
                    "clebby":data
                }
            }
        })
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='blebby')
        self.assertEquals(loaded['frebby']['clebby'], data)

        tmp = io.StringIO()
        mixed_data = [
            [1, 2, 3],
            "garbage",
            {"temps":[1., 2., 3.]}
            ]
        serializer.serialize(tmp, dict(mixed_data=mixed_data))
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='mixed_data')
        self.assertEquals(mixed_data, loaded)
```
#### <a name="JSONPseudoPickleSerialization">JSONPseudoPickleSerialization</a>
```python
    def test_JSONPseudoPickleSerialization(self):

        from McUtils.Numputils import SparseArray

        tmp = io.StringIO()
        serializer = JSONSerializer()

        data = SparseArray.from_diag([1, 2, 3, 4])

        serializer.serialize(tmp, data)
        tmp.seek(0)
        loaded = serializer.deserialize(tmp)

        self.assertTrue(np.allclose(loaded.asarray(), data.asarray()))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Serializers/JSONSerializer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Serializers/JSONSerializer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Serializers/JSONSerializer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/JSONSerializer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Serializers.py#L296?message=Update%20Docs)