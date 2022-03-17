## <a id="McUtils.Scaffolding.Schema.Schema">Schema</a>
An object that represents a schema that can be used to test
if an object matches that schema or not

### Properties and Methods
<a id="McUtils.Scaffolding.Schema.Schema.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, schema, optional_schema=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Schema.py#L13)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Schema.py#L13?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Schema.Schema.canonicalize_schema" class="docs-object-method">&nbsp;</a> 
```python
canonicalize_schema(schema): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Schema.py#L17)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Schema.py#L17?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Schema.Schema.validate" class="docs-object-method">&nbsp;</a> 
```python
validate(self, obj, throw=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Schema.py#L68)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Schema.py#L68?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Schema.py#L88)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Schema.py#L88?message=Update%20Docs)]
</div>

Converts `obj` into a plain `dict` representation
- `obj`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Schema.Schema.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Schema.py#L113)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Schema.py#L113?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Scaffolding/Schema/Schema.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Scaffolding/Schema/Schema.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Scaffolding/Schema/Schema.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Scaffolding/Schema/Schema.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Schema.py?message=Update%20Docs)