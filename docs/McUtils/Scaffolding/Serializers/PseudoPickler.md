## <a id="McUtils.Scaffolding.Serializers.PseudoPickler">PseudoPickler</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L20)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L20?message=Update%20Docs)]
</div>

A simple plugin to work _like_ pickle, in that it should
hopefully support serializing arbitrary python objects, but which
doesn't attempt to put stuff down to a single `bytearray`, instead
supporting objects with `to_state` and `from_state` methods by converting
them to more primitive serializble types like arrays, strings, numbers,
etc.
Falls back to naive pickling when necessary.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Scaffolding.Serializers.PseudoPickler.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, allow_pickle=False, protocol=1, b64encode=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L32)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L32?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Serializers.PseudoPickler.to_state" class="docs-object-method">&nbsp;</a> 
```python
to_state(self, obj, cache=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L121)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L121?message=Update%20Docs)]
</div>

Tries to extract state from `obj`, first through its `to_state`
        interface, but that failing by recursively walking the object
        tree
- `obj`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Serializers.PseudoPickler.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, obj, cache=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L152)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L152?message=Update%20Docs)]
</div>

Serializes an object first by checking for a `to_state`
        method, and that missing, by converting to primitive-ish types
        in a recursive strategy if the object passes `is_simple`, otherwise
        falling back to `pickle`
- `obj`: `Any`
    >object to be serialized
- `:returns`: `dict`
    >spec for the pseudo-pickled data

<a id="McUtils.Scaffolding.Serializers.PseudoPickler.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(self, spec): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Serializers.py#L196)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L196?message=Update%20Docs)]
</div>

Deserializes from an object spec, dispatching
        to regular pickle where necessary
- `object`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [Pseudopickle](#Pseudopickle)

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

#### <a name="Pseudopickle">Pseudopickle</a>
```python
    def test_Pseudopickle(self):

        from McUtils.Numputils import SparseArray

        pickler = PseudoPickler()
        spa = SparseArray.from_diag([1, 2, 3, 4])
        serial = pickler.serialize(spa)
        deserial = pickler.deserialize(serial)
        self.assertTrue(np.allclose(spa.asarray(), deserial.asarray()))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Serializers/PseudoPickler.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Serializers/PseudoPickler.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Serializers/PseudoPickler.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Serializers/PseudoPickler.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Serializers.py#L20?message=Update%20Docs)