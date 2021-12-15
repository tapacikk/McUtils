## <a id="McUtils.Scaffolding.Schema.Schema">Schema</a>
An object that represents a schema that can be used to test
if an object matches that schema or not

### Properties and Methods
<a id="McUtils.Scaffolding.Schema.Schema.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, schema, optional_schema=None): 
```

<a id="McUtils.Scaffolding.Schema.Schema.canonicalize_schema" class="docs-object-method">&nbsp;</a>
```python
canonicalize_schema(schema): 
```

<a id="McUtils.Scaffolding.Schema.Schema.validate" class="docs-object-method">&nbsp;</a>
```python
validate(self, obj, throw=True): 
```
Validates that `obj` matches the provided schema
        and throws an error if not
- `obj`: `Any`
    >No description...
- `throw`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Schema.Schema.to_dict" class="docs-object-method">&nbsp;</a>
```python
to_dict(self, obj, throw=True): 
```
Converts `obj` into a plain `dict` representation
- `obj`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Schema.Schema.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Schema/Schema.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Schema/Schema.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Schema/Schema.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Schema/Schema.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Schema.py?message=Update%20Docs)