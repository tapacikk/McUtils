## <a id="McUtils.Scaffolding.CLIs.CLI">CLI</a>
A representation of a command line interface
which layers simple command dispatching on the basic
ArgParse interface

### Properties and Methods
```python
extra_commands: dict
```
<a id="McUtils.Scaffolding.CLIs.CLI.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, name, description, *groups, cmd_name=None): 
```

- `name`: `str`
    >No description...
- `description`: `str`
    >No description...
- `cmd_name`: `str | None`
    >No description...
- `groups`: `Type[CommandGroup]`
    >No description...

<a id="McUtils.Scaffolding.CLIs.CLI.parse_group_command" class="docs-object-method">&nbsp;</a>
```python
parse_group_command(self): 
```
Parses a group and command argument (if possible) and prunes `sys.argv`
- `group`: `Any`
    >No description...
- `command`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.CLI.get_command" class="docs-object-method">&nbsp;</a>
```python
get_command(self): 
```

<a id="McUtils.Scaffolding.CLIs.CLI.get_group" class="docs-object-method">&nbsp;</a>
```python
get_group(self, grp): 
```

<a id="McUtils.Scaffolding.CLIs.CLI.run_command" class="docs-object-method">&nbsp;</a>
```python
run_command(self): 
```

<a id="McUtils.Scaffolding.CLIs.CLI.get_help" class="docs-object-method">&nbsp;</a>
```python
get_help(self): 
```
Gets the help string for the CLI
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.CLI.help" class="docs-object-method">&nbsp;</a>
```python
help(self, print_help=True): 
```

<a id="McUtils.Scaffolding.CLIs.CLI.run_parse" class="docs-object-method">&nbsp;</a>
```python
run_parse(self, parse, unknown): 
```
Provides a standard entry point to running stuff using the default CLI
- `parse`: `Any`
    >No description...
- `unknown`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.CLI.parse_toplevel_args" class="docs-object-method">&nbsp;</a>
```python
parse_toplevel_args(self): 
```
Parses out the top level flags that the program supports
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.CLI.run" class="docs-object-method">&nbsp;</a>
```python
run(self): 
```
Parses the arguments in `sys.argv` and dispatches to the approriate action.
        By default supports interactive sessions, running scripts, and abbreviated tracebacks.
- `:returns`: `_`
    >No description...

### Examples


