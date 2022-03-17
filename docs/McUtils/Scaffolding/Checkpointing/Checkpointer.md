## <a id="McUtils.Scaffolding.Checkpointing.Checkpointer">Checkpointer</a>
General purpose base class that allows checkpointing to be done easily and cleanly.
Intended to be a passable object that allows code to checkpoint easily.

### Properties and Methods
```python
default_extension: str
```
<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, checkpoint_file, allowed_keys=None, omitted_keys=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L27)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L27?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L37)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L37?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.extension_map" class="docs-object-method">&nbsp;</a> 
```python
extension_map(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L41)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L41?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.build_canonical" class="docs-object-method">&nbsp;</a> 
```python
build_canonical(checkpoint): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L48)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L48?message=Update%20Docs)]
</div>

Dispatches over types of objects to make a canonical checkpointer
        from the supplied data
- `checkpoint`: `None | str | Checkpoint | file | dict`
    >provides
- `:returns`: `Checkpointer`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.from_file" class="docs-object-method">&nbsp;</a> 
```python
from_file(file, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L81)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L81?message=Update%20Docs)]
</div>

Dispatch function to load from the appropriate file
- `file`: `str | File`
    >No description...
- `opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__enter__" class="docs-object-method">&nbsp;</a> 
```python
__enter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L108)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L108?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__exit__" class="docs-object-method">&nbsp;</a> 
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L113)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L113?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.is_open" class="docs-object-method">&nbsp;</a> 
```python
@property
is_open(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.stream" class="docs-object-method">&nbsp;</a> 
```python
@property
stream(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.open_checkpoint_file" class="docs-object-method">&nbsp;</a> 
```python
open_checkpoint_file(self, chk): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L128)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L128?message=Update%20Docs)]
</div>

Opens the passed `checkpoint_file` (if not already open)
- `chk`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.close_checkpoint_file" class="docs-object-method">&nbsp;</a> 
```python
close_checkpoint_file(self, stream): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L138)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L138?message=Update%20Docs)]
</div>

Closes the opened checkpointing stream
- `stream`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.save_parameter" class="docs-object-method">&nbsp;</a> 
```python
save_parameter(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L148)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L148?message=Update%20Docs)]
</div>

Saves a parameter to the checkpoint file
- `key`: `Any`
    >No description...
- `value`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.load_parameter" class="docs-object-method">&nbsp;</a> 
```python
load_parameter(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L160)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L160?message=Update%20Docs)]
</div>

Loads a parameter from the checkpoint file
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.check_allowed_key" class="docs-object-method">&nbsp;</a> 
```python
check_allowed_key(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L171)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L171?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__getitem__" class="docs-object-method">&nbsp;</a> 
```python
__getitem__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L185)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L185?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.__setitem__" class="docs-object-method">&nbsp;</a> 
```python
__setitem__(self, key, value): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L191)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L191?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Checkpointing.Checkpointer.keys" class="docs-object-method">&nbsp;</a> 
```python
keys(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Checkpointing.py#L198)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Checkpointing.py#L198?message=Update%20Docs)]
</div>

Returns the keys of currently checkpointed
        objects
- `:returns`: `_`
    >No description...




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [HDF5Problems](#HDF5Problems)

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

#### <a name="HDF5Problems">HDF5Problems</a>
```python
    def test_HDF5Problems(self):

        test = os.path.expanduser('~/Desktop/woof.hdf5')
        os.remove(test)
        checkpointer = Checkpointer.from_file(test)
        with checkpointer:
            checkpointer['why'] = [
                np.random.rand(1000, 5, 5),
                np.array(0),
                np.array(0)
            ]
        with checkpointer:
            checkpointer['why'] = [
                np.random.rand(1001, 5, 5),
                np.array(0),
                np.array(0)
            ]
        with checkpointer:
            checkpointer['why2'] = [
                np.random.rand(1001, 5, 5),
                np.array(0),
                np.array(0)
            ]

        with checkpointer as chk:
            self.assertEquals(list(chk.keys()), ['why', 'why2'])
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Scaffolding/Checkpointing/Checkpointer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Scaffolding/Checkpointing/Checkpointer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Scaffolding/Checkpointing/Checkpointer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Scaffolding/Checkpointing/Checkpointer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Checkpointing.py?message=Update%20Docs)