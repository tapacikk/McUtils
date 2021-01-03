## <a id="McUtils.McUtils.Scaffolding.Logging.Logger">Logger</a>
Defines a simple logger object to write log data to a file based on log levels.

### Properties and Methods
```python
default_verbosity: int
lookup: method
```
<a id="McUtils.McUtils.Scaffolding.Logging.Logger.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, log_file=None, verbosity=<LogLevel.All: 100>, padding='', newline='\n'): 
```

<a id="McUtils.McUtils.Scaffolding.Logging.Logger.block" class="docs-object-method">&nbsp;</a>
```python
block(self, **kwargs): 
```

<a id="McUtils.McUtils.Scaffolding.Logging.Logger.register" class="docs-object-method">&nbsp;</a>
```python
register(self, key): 
```
Registers the logger under the given key
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Logging.Logger.format_message" class="docs-object-method">&nbsp;</a>
```python
format_message(self, message, *params, **kwargs): 
```

<a id="McUtils.McUtils.Scaffolding.Logging.Logger.format_metainfo" class="docs-object-method">&nbsp;</a>
```python
format_metainfo(self, metainfo): 
```

<a id="McUtils.McUtils.Scaffolding.Logging.Logger.log_print" class="docs-object-method">&nbsp;</a>
```python
log_print(self, message, *params, print_options=None, padding=None, newline=None, metainfo=None, **kwargs): 
```

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

### Examples


