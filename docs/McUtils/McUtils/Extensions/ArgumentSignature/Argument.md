## <a id="McUtils.McUtils.Extensions.ArgumentSignature.Argument">Argument</a>
Defines a single Argument for a C-level caller to support default values, etc.
We use a two-pronged approach where we have a set of ArgumentType serializers/deserializers

### Properties and Methods
```python
arg_types: list
infer_type: method
infer_array_type: method
inferred_type_string: method
```
<a id="McUtils.McUtils.Extensions.ArgumentSignature.Argument.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, name, dtype, default=None): 
```

- `name`: `str`
    >the name of the argument
- `dtype`: `ArgumentType`
    >the type of the argument; at some point we'll support type inference...
- `default`: `Any`
    >the default value for the argument

<a id="McUtils.McUtils.Extensions.ArgumentSignature.Argument.cpp_signature" class="docs-object-method">&nbsp;</a>
```python
@property
cpp_signature(self): 
```

<a id="McUtils.McUtils.Extensions.ArgumentSignature.Argument.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

### Examples
