## <a id="McUtils.Extensions.ArgumentSignature.FunctionSignature">FunctionSignature</a>
Defines a function signature for a C-level caller.
To be used inside `SharedLibraryFunction` and things to manage the core interface.

### Properties and Methods
<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, name, *args, return_type=None): 
```

- `name`: `str`
    >the name of the function
- `args`: `Iterable[ArgumentType]`
    >the arguments passed to the function
- `return_type`: `ArgumentType | None`
    >the return type of the function

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.build_argument" class="docs-object-method">&nbsp;</a>
```python
build_argument(self, argtup, which=None): 
```
Converts an argument tuple into an Argument object
- `argtup`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.args" class="docs-object-method">&nbsp;</a>
```python
@property
args(self): 
```

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.return_type" class="docs-object-method">&nbsp;</a>
```python
@property
return_type(self): 
```

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.cpp_signature" class="docs-object-method">&nbsp;</a>
```python
@property
cpp_signature(self): 
```

<a id="McUtils.Extensions.ArgumentSignature.FunctionSignature.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Extensions/ArgumentSignature/FunctionSignature.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Extensions/ArgumentSignature.py?message=Update%20Docs)