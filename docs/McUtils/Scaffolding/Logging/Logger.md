## <a id="McUtils.Scaffolding.Logging.Logger">Logger</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L138)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L138?message=Update%20Docs)]
</div>

Defines a simple logger object to write log data to a file based on log levels.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
LogLevel: EnumMeta
default_verbosity: LogLevel
```
<a id="McUtils.Scaffolding.Logging.Logger.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, log_file=None, log_level=None, print_function=None, padding='', newline='\n'): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L147)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L147?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.Logger.to_state" class="docs-object-method">&nbsp;</a> 
```python
to_state(self, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L164)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L164?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.Logger.from_state" class="docs-object-method">&nbsp;</a> 
```python
from_state(state, serializer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L172)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L172?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.Logger.block" class="docs-object-method">&nbsp;</a> 
```python
block(self, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L176)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L176?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.Logger.register" class="docs-object-method">&nbsp;</a> 
```python
register(self, key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L179)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L179?message=Update%20Docs)]
</div>

Registers the logger under the given key
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Logging.Logger.lookup" class="docs-object-method">&nbsp;</a> 
```python
lookup(key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L188)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L188?message=Update%20Docs)]
</div>

Looks up a logger. Has the convenient, but potentially surprising
        behavior that if no logger is found a `NullLogger` is returned.
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Logging.Logger.preformat_keys" class="docs-object-method">&nbsp;</a> 
```python
preformat_keys(key_functions): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L206)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L206?message=Update%20Docs)]
</div>

Generates a closure that will take the supplied
        keys/function pairs and update them appropriately
- `key_functions`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Logging.Logger.format_message" class="docs-object-method">&nbsp;</a> 
```python
format_message(self, message, *params, preformatter=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L227)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L227?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.Logger.format_metainfo" class="docs-object-method">&nbsp;</a> 
```python
format_metainfo(self, metainfo): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L248)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L248?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.Logger.split_lines" class="docs-object-method">&nbsp;</a> 
```python
split_lines(obj): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L255)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L255?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.Logger.log_print" class="docs-object-method">&nbsp;</a> 
```python
log_print(self, message, *messrest, message_prepper=None, padding=None, newline=None, log_level=None, metainfo=None, print_function=None, print_options=None, sep=None, end=None, file=None, flush=None, preformatter=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L259)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L259?message=Update%20Docs)]
</div>


- `message`: `str | Iterable[str]`
    >message to print
- `params`: `Any`
    >No description...
- `print_options`: `Any`
    >options to be passed through to print
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Logging.Logger.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L346)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L346?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [BasicLogging](#BasicLogging)
- [InformedLogging](#InformedLogging)

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

#### <a name="BasicLogging">BasicLogging</a>
```python
    def test_BasicLogging(self):
        stdout = io.StringIO()
        logger = Logger(stdout)
        with logger.block(tag='Womp Womp'):
            logger.log_print('wompy dompy domp')

            logger.log_print('Some other useful info?')
            with logger.block(tag="Calling into subprogram"):
                logger.log_print('actually this is fake -_-')
                logger.log_print('took {timing:.5f}s', timing=121.01234)

            logger.log_print('I guess following up on that?')
            with logger.block(tag="Calling into subprogram"):
                logger.log_print('this is also fake! :yay:')
                logger.log_print('took {timing:.5f}s', timing=212.01234)

            logger.log_print('done for now; took {timing:.5f}s', timing=-1)

        with logger.block(tag='Surprise second block!'):
            logger.log_print('just kidding')
            with logger.block(tag="JK on that JK"):
                with logger.block(tag="Doubly nested block!"):
                    logger.log_print('woopy doopy doo bitchez')
                logger.log_print('(all views are entirely my own and do not reflect on my employer in any way)')

            logger.log_print('okay done for real; took {timing:.0f} years', timing=10000)

        with tmpf.NamedTemporaryFile(mode="w+b") as temp:
            log_dump = temp.name
        try:
            with open(log_dump, "w+") as dump:
                dump.write(stdout.getvalue())
            with LogParser(log_dump) as parser:
                blocks = list(parser.get_blocks())
                self.assertEquals(blocks[1].lines[1].lines[1], " (all views are entirely my own and do not reflect on my employer in any way)")
                self.assertEquals(blocks[1].lines[1].lines[0].tag, "Doubly nested block!")
        finally:
            os.remove(log_dump)
```
#### <a name="InformedLogging">InformedLogging</a>
```python
    def test_InformedLogging(self):
        import random

        with tmpf.NamedTemporaryFile(mode="w+b") as temp:
            log_dump = temp.name
        try:
            logger = Logger(log_dump)
            for i in range(100):
                with logger.block(tag="Step {}".format(i)):
                    logger.log_print("Did X")
                    logger.log_print("Did Y")
                    with logger.block(tag="Fake Call".format(i)):
                        logger.log_print("Took {timing:.5f}s", timing=random.random())

            number_puller = parsers.StringParser(parsers.Capturing(parsers.Number))
            with LogParser(log_dump) as parser:
                time_str = ""
                for block in parser.get_blocks(tag="Fake Call", level=1):
                    time_str += block.lines[0]
                timings = number_puller.parse_all(time_str).array
                self.assertEquals(len(timings), 100)
                self.assertGreater(np.average(timings), .35)
                self.assertLess(np.average(timings), .65)

            with LogParser(log_dump) as parser:
                time_str = ""
                for line in parser.get_lines(tag="Took ", level=1):
                    time_str += line
                timings = number_puller.parse_all(time_str).array
                self.assertEquals(len(timings), 100)
                self.assertGreater(np.average(timings), .35)
                self.assertLess(np.average(timings), .65)

        finally:
            os.remove(log_dump)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Logging/Logger.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Logging/Logger.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Logging/Logger.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Logging/Logger.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L138?message=Update%20Docs)