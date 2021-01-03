## <a id="McUtils.McUtils.Extensions.ArgumentSignature.ArgumentType">ArgumentType</a>
Defines a general purpose `ArgumentType` so that we can easily manage complicated type specs
The basic idea is to define a hierarchy of types that can then convert themselves down to
a `ctypes`-style spec as well as a C++ argument spec so that we can enable `SharedLibraryFunction`
to use either the basic `ctypes` FFI or a more efficient, but fragile system based off of extension modules.
This will be explicitly overridden by the `PrimitiveType`, `ArrayType` and `PointerType` subclasses that provide
the actual useable classes.
I'd really live to be integrate with what's in the `typing` module to be able to reuse that type-inference machinery

### Properties and Methods
<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArgumentType.ctypes_type" class="docs-object-method">&nbsp;</a>
```python
@property
ctypes_type(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArgumentType.cpp_type" class="docs-object-method">&nbsp;</a>
```python
@property
cpp_type(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArgumentType.types" class="docs-object-method">&nbsp;</a>
```python
@property
types(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArgumentType.dtypes" class="docs-object-method">&nbsp;</a>
```python
@property
dtypes(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArgumentType.isinstance" class="docs-object-method">&nbsp;</a>
```python
isinstance(self, arg): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.ArgumentType.cast" class="docs-object-method">&nbsp;</a>
```python
cast(self, arg): 
```

### Examples


