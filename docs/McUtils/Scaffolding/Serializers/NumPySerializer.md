## <a id="McUtils.Scaffolding.Serializers.NumPySerializer">NumPySerializer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L851)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L851?message=Update%20Docs)]
</div>

A serializer that implements NPZ dumps

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_extension: str
atomic_types: tuple
converter_dispatch: NoneType
dict_key_sep: str
```
<a id="McUtils.Scaffolding.Serializers.NumPySerializer.get_default_converters" class="docs-object-method">&nbsp;</a> 
```python
get_default_converters(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L861)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L861?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.get_converters" class="docs-object-method">&nbsp;</a> 
```python
get_converters(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L871)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L871?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.convert" class="docs-object-method">&nbsp;</a> 
```python
convert(self, data): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L946)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L946?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.deconvert" class="docs-object-method">&nbsp;</a> 
```python
deconvert(self, data, sep=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L970)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L970?message=Update%20Docs)]
</div>

Unflattens nested dictionary structures so that the original data
        can be recovered
- `data`: `Any`
    >No description...
- `sep`: `str | None`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, file, data, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L1002)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L1002?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.NumPySerializer.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(self, file, key=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L1010)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L1010?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [NumPySerialization](#NumPySerialization)

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

#### <a name="NumPySerialization">NumPySerialization</a>
```python
    def test_NumPySerialization(self):
        tmp = io.BytesIO()
        serializer = NumPySerializer()

        data = [1, 2, 3]
        serializer.serialize(tmp, data)
        tmp.seek(0)
        loaded = serializer.deserialize(tmp)
        self.assertEquals(loaded.tolist(), data)

        tmp = io.BytesIO()
        serializer.serialize(tmp, {
            "blebby": {
                "frebby": {
                    "clebby": data
                }
            }
        })
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='blebby')
        self.assertEquals(loaded['frebby']['clebby'].tolist(), data)

        tmp = io.BytesIO()
        mixed_data = [
            [1, 2, 3],
            "garbage",
            {"temps": [1., 2., 3.]}
        ]
        serializer.serialize(tmp, dict(mixed_data=mixed_data))
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='mixed_data')
        self.assertEquals(mixed_data, [
            loaded[0].tolist(),
            loaded[1].tolist(),
            {k: v.tolist() for k, v in loaded[2].items()}
        ])
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Serializers/NumPySerializer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Serializers/NumPySerializer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Serializers/NumPySerializer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/NumPySerializer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L851?message=Update%20Docs)