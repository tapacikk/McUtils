
## RegexPattern

A `RegexPattern` is a higher-level interface to work with the [regular expression](https://en.wikipedia.org/wiki/Regular_expression) (regex) string pattern matching language.
Python provides support for regular expressions through the [`re`](https://docs.python.org/3/library/re.html) module.
Being comfortable with regex is not a requirement for working with `RegexPattern` but will help explain some of the more confusing design decisions.

There are a bunch of different `RegexPattern` instances that cover different cases, e.g.

* `Word`: matches a string of characters that are generally considered _text_
* `PositiveInteger`: matches a string of characters that are only _digits_
* `Integer`: a `PositiveInteger` with and optional sign
* `Number`: matches `Integer.PositiveInteger`
* `VariableName`: matches a string of digits or text as the first character is a letter
* `Optional`: represents an _optional_ pattern to match

### Capturing/Named

When matching pieces of text it is also important to specify which pieces of text we would like to actually get back out.
For this there are two main `RegexPattern` instances.
The simplest one is `Capturing`.
This just specifies that we would like to capture a piece of text.
There is a slightly more sophisticated instance called `Named` which allows us to attach a _name_ to a group.

<div class="card in-out-block" markdown="1" id="Markdown_code">

```python
key_value_matcher = RegexPattern([Named(Word, "key"), "=", Named(Word, "value")])
print(key_value_matcher)
```

<div class="card-body out-block" markdown="1">

```lang-none
(?P<key>\w+)(?:=)(?P<value>\w+)
```

</div>
</div>

This can be used directly to pull info out of files

<div class="card in-out-block" markdown="1" id="Markdown_code">

```python
test_data = os.path.join(os.path.dirname(McUtils.__file__), 'ci', 'tests', 'TestData')
with open(os.path.join(test_data, 'water_OH_scan.log')) as log_dat:
    sample_data = log_dat.read()

matches = list(key_value_matcher.finditer(sample_data))
for match in matches[:5]:
    print(match.groupdict())
```

<div class="card-body out-block" markdown="1">

```python
{'key': '0', 'value': 'g09'}
{'key': 'Input', 'value': 'water_OH_scan'}
{'key': 'Output', 'value': 'water_OH_scan'}
{'key': 'Chk', 'value': 'water_OH_scan'}
{'key': 'NProc', 'value': '8'}
```

</div>
</div>

## StringParser

A more powerful interface than `RegexPattern` is through a `StringParser` instance.
This provides a wrapper on `RegexPattern` that handles the process of turning matches into `NumPy` arrays of the appropriate type.
The actual interface is quite simple, e.g. we can take our matcher from before and use it directly

<div class="card in-out-block" markdown="1" id="Markdown_code">

```python
key_vals = StringParser(key_value_matcher).parse_all(sample_data)
print(key_vals)
```

<div class="card-body out-block" markdown="1">

```python
StructuredTypeArray(shape=[(11493, 0), (11493, 0)], dtype=OrderedDict([('key', StructuredType(<class 'str'>, shape=(None,))), ('value', StructuredType(<class 'str'>, shape=(None,)))]))
```

</div>
</div>

This `StructuredTypeArray` is basically a version of `NumPy` [record arrays](https://numpy.org/doc/stable/reference/generated/numpy.recarray.html), 
but was written without knowing about them.
A smarter reimplementation of this portion of the parsing process would make use of `recarray` instead of this custom array type.

That said, getting the raw `ndarray` objects out is straight-forward

<div class="card in-out-block" markdown="1" id="Markdown_code">

```python
key_vals['key'].array
```

<div class="card-body out-block" markdown="1">

```python
array(['0', 'Input', 'Output', ..., 'State', 'RMSD', 'PG'], dtype='<U7')
```

</div>
</div>

NOTE: 90% of all bugs in the `StringParser` ecosystem will come from the design of `StructuredTypeArray`. 
The need to be efficient in data handling can lead to some difficult implementation details. 
As the data type has organically evolved it has become potentially tough to understand.
A reimplementation based on `recarray` would _potentially_ solve some issues.
{: .alert .alert-warning}

### Block Handlers

For efficiency sake, `StringParser` objects also provide a `block_handlers` argument (and handlers can be defined on `RegexPatterns` directly).
A handler is a function that can be applied to a parsed piece of text and should directly return a `NumPy` array so that it can be worked into the returned `StructuredTypeArray`.
The simplest handlers are already provided for convenience on `StringParser`, e.g. from `GaussianLogComponents.py`

```python
Named(
    Repeating(
        Capturing(Number),
        min = 3, max = 3,
        prefix=Optional(Whitespace),
        joiner = Whitespace
    ),
    "Coordinates", handler=StringParser.array_handler(dtype=float)
)
```

Here `StringParser.array_handler(dtype=float)` provides efficient parsing of data through `np.loadtxt` with a `float` as the target `dtype`.
We also see the `prefix` and `joiner` options to `RegexPattern` in action.