## <a id="McUtils.Scaffolding.CLIs.CLI">CLI</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L259)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L259?message=Update%20Docs)]
</div>

A representation of a command line interface
which layers simple command dispatching on the basic
ArgParse interface

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
extra_commands: dict
```
<a id="McUtils.Scaffolding.CLIs.CLI.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, name, description, *groups, cmd_name=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L265)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L265?message=Update%20Docs)]
</div>


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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L284)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L284?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L305)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L305?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.CLIs.CLI.get_group" class="docs-object-method">&nbsp;</a> 
```python
get_group(self, grp): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L314)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L314?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.CLIs.CLI.run_command" class="docs-object-method">&nbsp;</a> 
```python
run_command(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L322)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L322?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.CLIs.CLI.get_help" class="docs-object-method">&nbsp;</a> 
```python
get_help(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L332)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L332?message=Update%20Docs)]
</div>

Gets the help string for the CLI
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.CLI.help" class="docs-object-method">&nbsp;</a> 
```python
help(self, print_help=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L348)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L348?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.CLIs.CLI.run_parse" class="docs-object-method">&nbsp;</a> 
```python
run_parse(self, parse, unknown): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L355)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L355?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L425)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L425?message=Update%20Docs)]
</div>

Parses out the top level flags that the program supports
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.CLIs.CLI.run" class="docs-object-method">&nbsp;</a> 
```python
run(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/CLIs.py#L448)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L448?message=Update%20Docs)]
</div>

Parses the arguments in `sys.argv` and dispatches to the approriate action.
        By default supports interactive sessions, running scripts, and abbreviated tracebacks.
- `:returns`: `_`
    >No description...

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [CLI](#CLI)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
#### <a class="collapse-link" data-toggle="collapse" href="#test-setup">Setup</a> <a class="float-right" data-toggle="collapse" href="#test-setup"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="test-setup" markdown="1">

Before we can run our examples we should get a bit of setup out of the way.
Since these examples were harvested from the unit tests not all pieces
will be necessary for all situations.
```python
from Peeves.TestUtils import *
from McUtils.Scaffolding import *
import McUtils.Parsers as parsers
from unittest import TestCase
import numpy as np, io, os, sys, tempfile as tmpf
```

All tests are wrapped in a test class
```python
class ScaffoldingTests(TestCase):
    class DataHolderClass:
        def __init__(self, **keys):
            self.data = keys
        def to_state(self, serializer=None):
            return self.data
        @classmethod
        def from_state(cls, state, serializer=None):
            return cls(**state)
```

 </div>
</div>

#### <a name="CLI">CLI</a>
```python
    def test_CLI(self):
        import McUtils.Plots as plt
        class PlottingInterface(CommandGroup):
            _tag = "plot"
            @classmethod
            def random(cls, npts:int = 100, file:str = None):
                """Makes a random plot of however many points you want"""
                xy = np.random.rand(npts, npts)
                ploot = plt.ArrayPlot(xy)
                if file is None:
                    ploot.show()
                else:
                    ploot.savefig(file)
            @classmethod
            def contour(cls, npts: int = 100, file: str = None):
                """Makes a random contour plot of however many points you want"""
                xy = np.random.rand(npts, npts)
                ploot = plt.ListContourPlot(xy)
                if file is None:
                    ploot.show()
                else:
                    ploot.savefig(file)

        import McUtils.Data as data
        class DataInterface(CommandGroup):
            _tag = "data"
            @classmethod
            def mass(cls, elem:str):
                """Gets the mass for the passed element spec"""
                print(data.AtomData[elem]['Mass'])

        mccli = CLI(
            "McUtils",
            "defines a simple CLI interface to various bits of McUtils",
            PlottingInterface,
            DataInterface,
            cmd_name='mcutils'
        )
        print()

        with tmpf.NamedTemporaryFile() as out:
            argv = sys.argv
            try:
                sys.argv = ['mccli', '--help']
                mccli.run()

                sys.argv = ['mccli', 'plot', 'contour', '--npts=100']
                mccli.run()

                sys.argv = ['mccli', 'data', 'mass', 'T']
                mccli.run()
            finally:
                sys.argv = argv
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/CLIs/CLI.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/CLIs/CLI.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/CLIs/CLI.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/CLIs/CLI.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/CLIs.py#L259?message=Update%20Docs)