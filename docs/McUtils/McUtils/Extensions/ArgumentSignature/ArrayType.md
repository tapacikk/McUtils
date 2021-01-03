## <a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType">ArrayType</a>
Extends the basic `ArgumentType` spec to handle array types of possibly fixed size.
To start, we're only adding in proper support for numpy arrays.
Other flavors might come, but given the use case, it's unlikely.

### Properties and Methods
<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, base_type, shape=None): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.ctypes_type" class="docs-object-method">&nbsp;</a>
```python
@property
ctypes_type(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.cpp_type" class="docs-object-method">&nbsp;</a>
```python
@property
cpp_type(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.types" class="docs-object-method">&nbsp;</a>
```python
@property
types(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.dtypes" class="docs-object-method">&nbsp;</a>
```python
@property
dtypes(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.isinstance" class="docs-object-method">&nbsp;</a>
```python
isinstance(self, arg): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.cast" class="docs-object-method">&nbsp;</a>
```python
cast(self, arg): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArrayType.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

### Examples


