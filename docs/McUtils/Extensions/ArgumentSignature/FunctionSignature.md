## <a id="McUtils.Extensions.ArgumentSignature.FunctionSignature">FunctionSignature</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ArgumentSignature.py#L338)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature.py#L338?message=Update%20Docs)]
</div>

Defines a function signature for a C-level caller.
To be used inside `SharedLibraryFunction` and things to manage the core interface.







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 
<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, name, *args, return_type=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ArgumentSignature/FunctionSignature.py#L344)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature/FunctionSignature.py#L344?message=Update%20Docs)]
</div>

  - `name`: `str`
    > the name of the function
  - `args`: `Iterable[ArgumentType]`
    > the arguments passed to the function
  - `return_type`: `ArgumentType | None`
    > the return type of the function


<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.build_argument" class="docs-object-method">&nbsp;</a> 
```python
build_argument(self, argtup, which=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ArgumentSignature/FunctionSignature.py#L359)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature/FunctionSignature.py#L359?message=Update%20Docs)]
</div>
Converts an argument tuple into an Argument object
  - `argtup`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.args" class="docs-object-method">&nbsp;</a> 
```python
@property
args(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ArgumentSignature/FunctionSignature.py#L382)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature/FunctionSignature.py#L382?message=Update%20Docs)]
</div>


<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.return_type" class="docs-object-method">&nbsp;</a> 
```python
@property
return_type(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ArgumentSignature/FunctionSignature.py#L386)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature/FunctionSignature.py#L386?message=Update%20Docs)]
</div>


<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.cpp_signature" class="docs-object-method">&nbsp;</a> 
```python
@property
cpp_signature(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ArgumentSignature/FunctionSignature.py#L390)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature/FunctionSignature.py#L390?message=Update%20Docs)]
</div>


<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ArgumentSignature/FunctionSignature.py#L397)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature/FunctionSignature.py#L397?message=Update%20Docs)]
</div>
 </div>
</div>












---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Extensions/ArgumentSignature/FunctionSignature.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Extensions/ArgumentSignature/FunctionSignature.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Extensions/ArgumentSignature/FunctionSignature.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Extensions/ArgumentSignature/FunctionSignature.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ArgumentSignature.py#L338?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>