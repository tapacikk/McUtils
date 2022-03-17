## <a id="McUtils.Scaffolding.CLIs.Command">Command</a>
A holder for a command that just automates type handling &
that sort of thing

### Properties and Methods
<a id="McUtils.Scaffolding.CLIs.Command.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, name, method): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/CLIs.py#L19)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/CLIs.py#L19?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.CLIs.Command.get_help" class="docs-object-method">&nbsp;</a> 
```python
get_help(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/CLIs.py#L28)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/CLIs.py#L28?message=Update%20Docs)]
</div>

Gets single method help string
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.get_parse_dict" class="docs-object-method">&nbsp;</a> 
```python
get_parse_dict(*spec): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/CLIs.py#L91)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/CLIs.py#L91?message=Update%20Docs)]
</div>

Builds a parse spec to feed into an ArgumentParser later
- `spec`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.get_parse_spec" class="docs-object-method">&nbsp;</a> 
```python
get_parse_spec(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/CLIs.py#L128)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/CLIs.py#L128?message=Update%20Docs)]
</div>

Gets a parse spec that can be fed to ArgumentParser
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.parse" class="docs-object-method">&nbsp;</a> 
```python
parse(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/CLIs.py#L160)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/CLIs.py#L160?message=Update%20Docs)]
</div>

Generates a parse spec, builds an ArgumentParser, and parses the arguments
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.__call__" class="docs-object-method">&nbsp;</a> 
```python
__call__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/CLIs.py#L170)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/CLIs.py#L170?message=Update%20Docs)]
</div>

Parse argv and call bound method
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Scaffolding/CLIs/Command.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Scaffolding/CLIs/Command.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Scaffolding/CLIs/Command.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Scaffolding/CLIs/Command.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/CLIs.py?message=Update%20Docs)