## <a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType">PrimitiveType</a>
Defines a general purpose ArgumentType so that we can easily manage complicated type specs
The basic idea is to define a hierarchy of types that can then convert themselves down to
a `ctypes`-style spec as well as a C++ argument spec so that we can enable `SharedLibraryFunction`
to use either the basic `ctypes` FFI or a more efficient, but fragile system based off of extension modules

### Properties and Methods
<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, name, ctypes_spec, cpp_spec, capi_spec, python_types, numpy_dtypes, serializer, deserializer): 
```

- `name`: `str`
    >argument name (e.g. 'double')
- `ctypes_spec`: `Any`
    >the ctypes data-type that arguments of this type would be converted to
- `cpp_spec`: `str`
    >the C++ spec for this type (as a string)
- `capi_spec`: `str`
    >the python C-API string for use in `Py_BuildValue`
- `python_types`: `Iterable[type]`
    >the python types that this argument maps onto
- `numpy_dtypes`: `Iterable[np.dtype]`
    >the numpy dtypes that this argument maps onto
- `serializer`: `Callable`
    >a serializer for converting this object into a byte-stream
- `deserializer`: `Callable`
    >a deserializer for converting the byte-stream into a C-level object

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.name" class="docs-object-method">&nbsp;</a>
```python
@property
name(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.ctypes_type" class="docs-object-method">&nbsp;</a>
```python
@property
ctypes_type(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.cpp_type" class="docs-object-method">&nbsp;</a>
```python
@property
cpp_type(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.types" class="docs-object-method">&nbsp;</a>
```python
@property
types(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.dtypes" class="docs-object-method">&nbsp;</a>
```python
@property
dtypes(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.isinstance" class="docs-object-method">&nbsp;</a>
```python
isinstance(self, arg): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.cast" class="docs-object-method">&nbsp;</a>
```python
cast(self, arg): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.PrimitiveType.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

### Examples
