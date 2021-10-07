## <a id="McUtils.Scaffolding.CLIs.Command">Command</a>
A holder for a command that just automates type handling &
that sort of thing

### Properties and Methods
<a id="McUtils.Scaffolding.CLIs.Command.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, name, method): 
```

<a id="McUtils.Scaffolding.CLIs.Command.get_help" class="docs-object-method">&nbsp;</a>
```python
get_help(self): 
```
Gets single method help string
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.get_parse_dict" class="docs-object-method">&nbsp;</a>
```python
get_parse_dict(*spec): 
```
Builds a parse spec to feed into an ArgumentParser later
- `spec`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.get_parse_spec" class="docs-object-method">&nbsp;</a>
```python
get_parse_spec(self): 
```
Gets a parse spec that can be fed to ArgumentParser
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.parse" class="docs-object-method">&nbsp;</a>
```python
parse(self): 
```
Generates a parse spec, builds an ArgumentParser, and parses the arguments
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.Command.__call__" class="docs-object-method">&nbsp;</a>
```python
__call__(self): 
```
Parse argv and call bound method
- `:returns`: `_`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/CLIs/Command.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/CLIs/Command.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/CLIs/Command.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/CLIs/Command.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/CLIs.py?message=Update%20Docs)