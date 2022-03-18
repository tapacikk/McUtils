## <a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction">SharedLibraryFunction</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L15)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L15?message=Update%20Docs)]
</div>

An object that provides a way to call into a shared library function

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
InDir: type
```
<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, shared_library, signature, docstring=None, call_directory=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L20)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L20?message=Update%20Docs)]
</div>


- `shared_library`: `str |`
    >the path to the shared library file you want to use
- `function_signature`: `FunctionSignature`
    >the signature of the function to load
- `call_directory`: `str`
    >the directory for calling
- `docstring`: `str`
    >the docstring for the function

<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.lib" class="docs-object-method">&nbsp;</a> 
```python
@property
lib(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.initialize" class="docs-object-method">&nbsp;</a> 
```python
initialize(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L70)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L70?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.doc" class="docs-object-method">&nbsp;</a> 
```python
doc(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L78)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L78?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L80)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L80?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.call" class="docs-object-method">&nbsp;</a> 
```python
call(self, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L86)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L86?message=Update%20Docs)]
</div>

Calls the function we loaded.
        This will be parallelized out to handle more complicated usages.
- `kwargs`: `Any`
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

- [SOSig](#SOSig)

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

#### <a name="SOSig">SOSig</a>
```python
    def test_SOSig(self):
        lib_file = TestManager.test_data('libmbpol.so')
        mbpol = SharedLibraryFunction(lib_file,
                                      FunctionSignature(
                                          "calcpot_",
                                          Argument("nw", PointerType(IntType)),
                                          Argument("num_2", PointerType(RealType)),
                                          Argument("coords", ArrayType(RealType))
                                      )
                                      )
        self.assertTrue(
            "SharedLibraryFunction(FunctionSignature(calcpot_(Argument('nw', PointerType(PrimitiveType(int)))" in repr(
                mbpol))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L15?message=Update%20Docs)