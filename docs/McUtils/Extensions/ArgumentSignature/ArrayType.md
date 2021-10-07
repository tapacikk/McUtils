## <a id="McUtils.Extensions.ArgumentSignature.ArrayType">ArrayType</a>
Extends the basic `ArgumentType` spec to handle array types of possibly fixed size.
To start, we're only adding in proper support for numpy arrays.
Other flavors might come, but given the use case, it's unlikely.

### Properties and Methods
<a id="McUtils.Extensions.ArgumentSignature.ArrayType.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, base_type, shape=None): 
```

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.ctypes_type" class="docs-object-method">&nbsp;</a>
```python
@property
ctypes_type(self): 
```

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.cpp_type" class="docs-object-method">&nbsp;</a>
```python
@property
cpp_type(self): 
```

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.types" class="docs-object-method">&nbsp;</a>
```python
@property
types(self): 
```

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.dtypes" class="docs-object-method">&nbsp;</a>
```python
@property
dtypes(self): 
```

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.isinstance" class="docs-object-method">&nbsp;</a>
```python
isinstance(self, arg): 
```

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.cast" class="docs-object-method">&nbsp;</a>
```python
cast(self, arg): 
```

<a id="McUtils.Extensions.ArgumentSignature.ArrayType.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Extensions/ArgumentSignature/ArrayType.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Extensions/ArgumentSignature/ArrayType.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Extensions/ArgumentSignature/ArrayType.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Extensions/ArgumentSignature/ArrayType.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Extensions/ArgumentSignature.py?message=Update%20Docs)