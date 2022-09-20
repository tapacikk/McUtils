## <a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction">SharedLibraryFunction</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager.py#L15)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L15?message=Update%20Docs)]
</div>

An object that provides a way to call into a shared library function







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 ```python
InDir: InDir
```
<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, shared_library, signature, docstring=None, call_directory=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L20)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L20?message=Update%20Docs)]
</div>

  - `shared_library`: `str |`
    > the path to the shared library file you want to use
  - `function_signature`: `FunctionSignature`
    > the signature of the function to load
  - `call_directory`: `str`
    > the directory for calling
  - `docstring`: `str`
    > the docstring for the function


<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.lib" class="docs-object-method">&nbsp;</a> 
```python
@property
lib(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L63)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L63?message=Update%20Docs)]
</div>


<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.initialize" class="docs-object-method">&nbsp;</a> 
```python
initialize(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L70)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L70?message=Update%20Docs)]
</div>


<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.doc" class="docs-object-method">&nbsp;</a> 
```python
doc(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L78)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L78?message=Update%20Docs)]
</div>


<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L80)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L80?message=Update%20Docs)]
</div>


<a id="McUtils.Extensions.SharedLibraryManager.SharedLibraryFunction.call" class="docs-object-method">&nbsp;</a> 
```python
call(self, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L86)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager/SharedLibraryFunction.py#L86?message=Update%20Docs)]
</div>
Calls the function we loaded.
This will be parallelized out to handle more complicated usages.
  - `kwargs`: `Any`
    > 
  - `:returns`: `_`
    >
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Extensions/SharedLibraryManager/SharedLibraryFunction.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/SharedLibraryManager.py#L15?message=Update%20Docs)   
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