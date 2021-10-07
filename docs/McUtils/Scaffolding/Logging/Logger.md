## <a id="McUtils.Scaffolding.Logging.Logger">Logger</a>
Defines a simple logger object to write log data to a file based on log levels.

### Properties and Methods
```python
LogLevel: EnumMeta
default_verbosity: LogLevel
```
<a id="McUtils.Scaffolding.Logging.Logger.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, log_file=None, log_level=None, print_function=None, padding='', newline='\n'): 
```

<a id="McUtils.Scaffolding.Logging.Logger.block" class="docs-object-method">&nbsp;</a>
```python
block(self, **kwargs): 
```

<a id="McUtils.Scaffolding.Logging.Logger.register" class="docs-object-method">&nbsp;</a>
```python
register(self, key): 
```
Registers the logger under the given key
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Logging.Logger.lookup" class="docs-object-method">&nbsp;</a>
```python
lookup(key): 
```
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

<a id="McUtils.Scaffolding.Logging.Logger.format_metainfo" class="docs-object-method">&nbsp;</a>
```python
format_metainfo(self, metainfo): 
```

<a id="McUtils.Scaffolding.Logging.Logger.split_lines" class="docs-object-method">&nbsp;</a>
```python
split_lines(obj): 
```

<a id="McUtils.Scaffolding.Logging.Logger.log_print" class="docs-object-method">&nbsp;</a>
```python
log_print(self, message, *messrest, message_prepper=None, padding=None, newline=None, log_level=None, metainfo=None, print_function=None, print_options=None, sep=None, end=None, file=None, flush=None, preformatter=None, **kwargs): 
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

<a id="McUtils.Scaffolding.Logging.Logger.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

### Examples


