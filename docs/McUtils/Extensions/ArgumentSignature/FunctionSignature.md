## <a id="McUtils.Extensions.ArgumentSignature.FunctionSignature">FunctionSignature</a>
Defines a function signature for a C-level caller.
To be used inside `SharedLibraryFunction` and things to manage the core interface.

### Properties and Methods
<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, name, *args, return_type=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L344)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L344?message=Update%20Docs)]
</div>


- `name`: `str`
    >the name of the function
- `args`: `Iterable[ArgumentType]`
    >the arguments passed to the function
- `return_type`: `ArgumentType | None`
    >the return type of the function

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.build_argument" class="docs-object-method">&nbsp;</a> 
```python
build_argument(self, argtup, which=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L359)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L359?message=Update%20Docs)]
</div>

Converts an argument tuple into an Argument object
- `argtup`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.args" class="docs-object-method">&nbsp;</a> 
```python
@property
args(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.return_type" class="docs-object-method">&nbsp;</a> 
```python
@property
return_type(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.cpp_signature" class="docs-object-method">&nbsp;</a> 
```python
@property
cpp_signature(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L397)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L397?message=Update%20Docs)]
</div>




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [BasicTypeSig](#BasicTypeSig)

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
from unittest import TestCase
from McUtils.Extensions import *
import sys, os, numpy as np
```

All tests are wrapped in a test class
```python
class ExtensionsTests(TestCase):
```

 </div>
</div>

#### <a name="BasicTypeSig">BasicTypeSig</a>
```python
    def test_BasicTypeSig(self):
        sig = FunctionSignature(
            "my_func",
            Argument("num_1", RealType),
            Argument("num_2", RealType, default=5),
            Argument("some_int", IntType)
        )
        self.assertEquals(sig.cpp_signature, "void my_func(double num_1, double num_2, int some_int)")
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Extensions/ArgumentSignature.py?message=Update%20Docs)