## <a id="McUtils.Extensions.CLoader.CLoader">CLoader</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L8)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L8?message=Update%20Docs)]
</div>

A general loader for C++ extensions to python, based off of the kind of thing that I have had to do multiple times

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Extensions.CLoader.CLoader.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, lib_name, lib_dir, load_path=None, src_ext='src', description='An extension module', version='1.0.0', include_dirs=None, runtime_dirs=None, linked_libs=None, macros=None, extra_link_args=None, extra_compile_args=None, extra_objects=None, source_files=None, build_script=None, requires_make=False, out_dir=None, cleanup_build=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L13)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L13?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.CLoader.CLoader.load" class="docs-object-method">&nbsp;</a> 
```python
load(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L56)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L56?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.CLoader.CLoader.find_extension" class="docs-object-method">&nbsp;</a> 
```python
find_extension(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L75)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L75?message=Update%20Docs)]
</div>

Tries to find the extension in the top-level directory
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.CLoader.CLoader.compile_extension" class="docs-object-method">&nbsp;</a> 
```python
compile_extension(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L88)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L88?message=Update%20Docs)]
</div>

Compiles and loads a C++ extension
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.CLoader.CLoader.src_dir" class="docs-object-method">&nbsp;</a> 
```python
@property
src_dir(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.CLoader.CLoader.lib_lib_dir" class="docs-object-method">&nbsp;</a> 
```python
@property
lib_lib_dir(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.CLoader.CLoader.get_extension" class="docs-object-method">&nbsp;</a> 
```python
get_extension(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L108)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L108?message=Update%20Docs)]
</div>

Gets the Extension module to be compiled
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.CLoader.CLoader.custom_make" class="docs-object-method">&nbsp;</a> 
```python
custom_make(self, make_file, make_dir): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L144)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L144?message=Update%20Docs)]
</div>

A way to call a custom make file either for building the helper lib or for building the proper lib
- `make_file`: `Any`
    >No description...
- `make_dir`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.CLoader.CLoader.make_required_libs" class="docs-object-method">&nbsp;</a> 
```python
make_required_libs(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L247)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L247?message=Update%20Docs)]
</div>

Makes any libs required by the current one
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.CLoader.CLoader.build_lib" class="docs-object-method">&nbsp;</a> 
```python
build_lib(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L261)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L261?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.CLoader.CLoader.locate_lib" class="docs-object-method">&nbsp;</a> 
```python
locate_lib(self, root=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L287)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L287?message=Update%20Docs)]
</div>

Tries to locate the build library file (if it exists)
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.CLoader.CLoader.cleanup" class="docs-object-method">&nbsp;</a> 
```python
cleanup(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/CLoader.py#L320)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L320?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Extensions/CLoader/CLoader.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Extensions/CLoader/CLoader.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Extensions/CLoader/CLoader.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Extensions/CLoader/CLoader.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/CLoader.py#L8?message=Update%20Docs)