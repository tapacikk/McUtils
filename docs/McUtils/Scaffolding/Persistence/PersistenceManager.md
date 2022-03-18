## <a id="McUtils.Scaffolding.Persistence.PersistenceManager">PersistenceManager</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L74)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L74?message=Update%20Docs)]
</div>

Defines a manager that can load configuration data from a directory
or, maybe in the future, a SQL database or similar.
Requires class that supports `from_config` to load and `to_config` to save.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, cls, persistence_loc=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L80)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L80?message=Update%20Docs)]
</div>


- `cls`: `type`
    >No description...
- `persistence_loc`: `str | None`
    >location from which to load/save objects

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.obj_loc" class="docs-object-method">&nbsp;</a> 
```python
obj_loc(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L98)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L98?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.load_config" class="docs-object-method">&nbsp;</a> 
```python
load_config(self, key, make_new=False, init=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L101)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L101?message=Update%20Docs)]
</div>

Loads the config for the persistent structure named `key`
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.new_config" class="docs-object-method">&nbsp;</a> 
```python
new_config(self, key, init=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L119)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L119?message=Update%20Docs)]
</div>

Creates a new space and config for the persistent structure named `key`
- `key`: `str`
    >name for job
- `init`: `str | dict | None`
    >initial parameters
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.contains" class="docs-object-method">&nbsp;</a> 
```python
contains(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L163)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L163?message=Update%20Docs)]
</div>

Checks if `key` is a supported persistent structure
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.load" class="docs-object-method">&nbsp;</a> 
```python
load(self, key, make_new=False, strict=True, init=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L174)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L174?message=Update%20Docs)]
</div>

Loads the persistent structure named `key`
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.save" class="docs-object-method">&nbsp;</a> 
```python
save(self, obj): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Persistence.py#L196)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L196?message=Update%20Docs)]
</div>

Saves requisite config data for a structure
- `obj`: `Any`
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

- [Persistence](#Persistence)

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

#### <a name="Persistence">Persistence</a>
```python
    def test_Persistence(self):
        persist_dir = TestManager.test_data("persistence_tests")

        class PersistentMock:
            """
            A fake object that supports the persistence interface we defined
            """

            def __init__(self, name, sample_val):
                self.name = name
                self.val = sample_val
            @classmethod
            def from_config(cls, name="wat", sample_val=None):
                return cls(name, sample_val)

        manager = PersistenceManager(PersistentMock, persist_dir)

        obj = manager.load("obj1", strict=False)

        self.assertEquals(obj.val, 'test_val')
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Persistence/PersistenceManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Persistence/PersistenceManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Persistence/PersistenceManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Persistence/PersistenceManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Persistence.py#L74?message=Update%20Docs)