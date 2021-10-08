
## FChk Parsing

Gaussian `.fchk` files have a set structure which looks roughly like
```lang-none
key    data_type     data_size
 data
```
This allows us to provide a complete parser for any `key`.
The actual parser is a subclass of `[Parsers.FileStreamReader]`(../Parsers/) called `GaussianFchkReader`.

The syntax to parse is straightforward

```python
target_keys = {"Current cartesian coordinates", "Numerical dipole derivatives"}
with GaussianFchkReader("/path/to/output.log") as parser:
    res = parser.parse(target_keys)
```

and to access properties you will pull them from the dict, `res`

```python
my_coords = res["Current cartesian coordinates"]
```


## Log Parsing

Gaussian `.log` files are totally unstructured (and a bit of a disaster). 
This means we need to write custom parsing logic for every field we might want.
The basic supported formats are defined in `GaussianLogComponents.py`. 
The actual parser is a subclass of `[Parsers.FileStreamReader]`(../Parsers/) called `GaussianLogReader`.

The syntax to parse is straightforward

```python
target_keys = {"StandardCartesianCoordinates", "DipoleMoments"}
with GaussianLogReader("/path/to/output.log") as parser:
    res = parser.parse(target_keys)
```

and to access properties you will pull them from the dict, `res`

```python
my_coords = res["StandardCartesianCoordinates"]
```

### Adding New Parsing Fields

New parse fields can be added by registering a property on `GaussianLogComponents`. 
Each field is defined as a dict like

```python
GaussianLogComponents["Name"] = {
    "description" : string, # used for docmenting what we have
    "tag_start"   : start_tag, # starting delimeter for a block
    "tag_end"     : end_tag, # ending delimiter for a block None means apply the parser upon tag_start
    "parser"      : parser, # function that'll parse the returned list of blocks (for "List") or block (for "Single")
    "mode"        : mode # "List" or "Single"
}
```

The `mode` argument specifies whether all blocks should be matched first and send to the `parser` (`"List"`) or if they should be fed in one-by-one `"Single"`.
This often provides a tradeoff between parsing efficiency and memory efficiency.

The `parser` can be any function, but commonly is built off of a `[Parsers.StringParser]`(../Parsers/). 
See the documentation for `StringParser` for more.

You can add to `GaussianLogComponents` at runtime.
Not all changes need to be integrated directly into the file.
{: .alert .alert-info}

## GJF Setup

Support is also provided for the automatic generation of Gaussian job files (`.gjf`) through the `GaussianJob` class.