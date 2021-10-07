## <a id="McUtils.Parsers.StringParser.StringParser">StringParser</a>
A convenience class that makes it easy to pull blocks out of strings and whatnot

### Properties and Methods
```python
MatchIterator: type
```
<a id="McUtils.Parsers.StringParser.StringParser.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, regex): 
```

<a id="McUtils.Parsers.StringParser.StringParser.parse" class="docs-object-method">&nbsp;</a>
```python
parse(self, txt, regex=None, block_handlers=None, dtypes=None, out=None): 
```
Finds a single match for the and applies parsers for the specified regex in txt
- `txt`: `str`
    >a chunk of text to be matched
- `regex`: `RegexPattern`
    >the regex to match in _txt_
- `block_handlers`: `iterable[callable] | OrderedDict[str: callable]`
    >handlers for the matched blocks in _regex_ -- usually comes from _regex_
- `dtypes`: `iterable[type | StructuredType] | OrderedDict[str: type | StructuredType]`
    >the types of the data that we expect to match -- usually comes from _regex_
- `out`: `None | StructuredTypeArray | iterable[StructuredTypeArray] | OrderedDict[str: StructuredTypeArray]`
    >where to place the parsed out data -- usually comes from _regex_
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.StringParser.StringParser.parse_all" class="docs-object-method">&nbsp;</a>
```python
parse_all(self, txt, regex=None, num_results=None, block_handlers=None, dtypes=None, out=None): 
```

<a id="McUtils.Parsers.StringParser.StringParser.parse_iter" class="docs-object-method">&nbsp;</a>
```python
parse_iter(self, txt, regex=None, num_results=None, block_handlers=None, dtypes=None): 
```

<a id="McUtils.Parsers.StringParser.StringParser.get_regex_block_handlers" class="docs-object-method">&nbsp;</a>
```python
get_regex_block_handlers(regex): 
```
Uses the uncompiled RegexPattern to determine what blocks exist and what handlers they should use
- `regex`: `RegexPattern`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.StringParser.StringParser.get_regex_dtypes" class="docs-object-method">&nbsp;</a>
```python
get_regex_dtypes(regex): 
```
Uses the uncompiled RegexPattern to determine which StructuredTypes to return
- `regex`: `RegexPattern`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.StringParser.StringParser.handler_method" class="docs-object-method">&nbsp;</a>
```python
handler_method(method): 
```
Turns a regular function into a handler method by adding in (and ignoring) the array argument
- `method`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.StringParser.StringParser.load_array" class="docs-object-method">&nbsp;</a>
```python
load_array(data, dtype='float'): 
```

<a id="McUtils.Parsers.StringParser.StringParser.to_array" class="docs-object-method">&nbsp;</a>
```python
to_array(data, array=None, append=False, dtype='float', shape=None, pre=None): 
```
A method to take a string or iterable of strings and quickly dump it to a NumPy array of the right dtype (if it can be cast as one)
- `data`: `Any`
    >No description...
- `dtype`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parsers.StringParser.StringParser.array_handler" class="docs-object-method">&nbsp;</a>
```python
array_handler(array=None, append=False, dtype='float', shape=None, pre=None): 
```
Returns a handler that uses to_array
- `dtype`: `Any`
    >No description...
- `array`: `Any`
    >No description...
- `shape`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parsers/StringParser/StringParser.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parsers/StringParser/StringParser.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parsers/StringParser/StringParser.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parsers/StringParser/StringParser.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parsers/StringParser.py?message=Update%20Docs)