## <a id="McUtils.Extensions.ArgumentSignature.ArrayType">ArrayType</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L126)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L126?message=Update%20Docs)]
</div>

Extends the basic `ArgumentType` spec to handle array types of possibly fixed size.
To start, we're only adding in proper support for numpy arrays.
Other flavors might come, but given the use case, it's unlikely.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, base_type, shape=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L132)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L132?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.ctypes_type" class="docs-object-method">&nbsp;</a> 
```python
@property
ctypes_type(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.cpp_type" class="docs-object-method">&nbsp;</a> 
```python
@property
cpp_type(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.types" class="docs-object-method">&nbsp;</a> 
```python
@property
types(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.dtypes" class="docs-object-method">&nbsp;</a> 
```python
@property
dtypes(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.isinstance" class="docs-object-method">&nbsp;</a> 
```python
isinstance(self, arg): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L150)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L150?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.cast" class="docs-object-method">&nbsp;</a> 
```python
cast(self, arg): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L152)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L152?message=Update%20Docs)]
</div>

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ArgumentSignature.py#L154)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L154?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Extensions/ArgumentSignature/ArrayType.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Extensions/ArgumentSignature/ArrayType.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Extensions/ArgumentSignature/ArrayType.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Extensions/ArgumentSignature/ArrayType.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ArgumentSignature.py#L126?message=Update%20Docs)