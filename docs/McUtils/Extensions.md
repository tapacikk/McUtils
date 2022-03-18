# <a id="McUtils.Extensions">McUtils.Extensions</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions)]
</div>
    
A package for managing extension modules.
The existing `ExtensionLoader` will be moving here, and will be supplemented by classes for dealing with compiled extensions

<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[CLoader](Extensions/CLoader/CLoader.md)   
</div>
   <div class="col" markdown="1">
[ModuleLoader](Extensions/ModuleLoader/ModuleLoader.md)   
</div>
   <div class="col" markdown="1">
[ArgumentType](Extensions/ArgumentSignature/ArgumentType.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[ArrayType](Extensions/ArgumentSignature/ArrayType.md)   
</div>
   <div class="col" markdown="1">
[PointerType](Extensions/ArgumentSignature/PointerType.md)   
</div>
   <div class="col" markdown="1">
[PrimitiveType](Extensions/ArgumentSignature/PrimitiveType.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[PrimitiveType](Extensions/ArgumentSignature/PrimitiveType.md)   
</div>
   <div class="col" markdown="1">
[PrimitiveType](Extensions/ArgumentSignature/PrimitiveType.md)   
</div>
   <div class="col" markdown="1">
[Argument](Extensions/ArgumentSignature/Argument.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[FunctionSignature](Extensions/ArgumentSignature/FunctionSignature.md)   
</div>
   <div class="col" markdown="1">
[SharedLibraryFunction](Extensions/SharedLibraryManager/SharedLibraryFunction.md)   
</div>
</div>
</div>




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [BasicTypeSig](#BasicTypeSig)
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

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Extensions.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Extensions.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Extensions.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Extensions.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/__init__.py?message=Update%20Docs)