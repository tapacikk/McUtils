## <a id="McUtils.McUtils.Scaffolding.Logging.LogParser">LogParser</a>
A parser that will take a log file and stream it as a series of blocks

### Properties and Methods
```python
LogBlockParser: type
```
<a id="McUtils.McUtils.Scaffolding.Logging.LogParser.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, file, block_settings=None, block_level_padding=None, **kwargs): 
```

<a id="McUtils.McUtils.Scaffolding.Logging.LogParser.get_block_settings" class="docs-object-method">&nbsp;</a>
```python
get_block_settings(self, block_level): 
```

<a id="McUtils.McUtils.Scaffolding.Logging.LogParser.get_block" class="docs-object-method">&nbsp;</a>
```python
get_block(self, level=0, tag=None): 
```

- `level`: `Any`
    >No description...
- `tag`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Logging.LogParser.get_line" class="docs-object-method">&nbsp;</a>
```python
get_line(self, level=0, tag=None): 
```

- `level`: `Any`
    >No description...
- `tag`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Logging.LogParser.get_blocks" class="docs-object-method">&nbsp;</a>
```python
get_blocks(self, tag=None, level=0): 
```

<a id="McUtils.McUtils.Scaffolding.Logging.LogParser.get_lines" class="docs-object-method">&nbsp;</a>
```python
get_lines(self, tag=None, level=0): 
```

### Examples


