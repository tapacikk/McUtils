## <a id="McUtils.Extensions.ArgumentSignature.Argument">Argument</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L272)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L272?message=Update%20Docs)]
</div>

Defines a single Argument for a C-level caller to support default values, etc.
We use a two-pronged approach where we have a set of ArgumentType serializers/deserializers

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
arg_types: list
```
<a id="McUtils.Extensions.ArgumentSignature.Argument.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, name, dtype, default=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L284)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L284?message=Update%20Docs)]
</div>


- `name`: `str`
    >the name of the argument
- `dtype`: `ArgumentType`
    >the type of the argument; at some point we'll support type inference...
- `default`: `Any`
    >the default value for the argument

<a id="McUtils.Extensions.ArgumentSignature.Argument.infer_type" class="docs-object-method">&nbsp;</a> 
```python
infer_type(arg): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L297)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L297?message=Update%20Docs)]
</div>

Infers the type of an argument
- `arg`: `ArgumentType | str | type | ctypes type`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.ArgumentSignature.Argument.infer_array_type" class="docs-object-method">&nbsp;</a> 
```python
infer_array_type(argstr): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L314)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L314?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.Argument.inferred_type_string" class="docs-object-method">&nbsp;</a> 
```python
inferred_type_string(arg): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L318)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L318?message=Update%20Docs)]
</div>

returns a type string for the inferred type

<a id="McUtils.Extensions.ArgumentSignature.Argument.cpp_signature" class="docs-object-method">&nbsp;</a> 
```python
@property
cpp_signature(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.Argument.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L331)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L331?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Extensions/ArgumentSignature/Argument.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Extensions/ArgumentSignature/Argument.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Extensions/ArgumentSignature/Argument.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Extensions/ArgumentSignature/Argument.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L272?message=Update%20Docs)