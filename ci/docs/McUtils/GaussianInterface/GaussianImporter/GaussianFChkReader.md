## <a id="McUtils.GaussianInterface.GaussianImporter.GaussianFChkReader">GaussianFChkReader</a>
Implements a stream based reader for a Gaussian .fchk file. Pretty generall I think. Should be robust-ish.
One place to change things up is convenient parsers for specific commonly pulled parts of the fchk

### Properties and Methods
```python
GaussianFChkReaderException: type
registered_components: dict
common_names: dict
to_common_name: dict
fchk_re_pattern: str
fchk_re: Pattern
```
<a id="McUtils.GaussianInterface.GaussianImporter.GaussianFChkReader.read_header" class="docs-object-method">&nbsp;</a>
```python
read_header(self): 
```
Reads the header and skips the stream to where we want to be
- `:returns`: `str`
    >the header

<a id="McUtils.GaussianInterface.GaussianImporter.GaussianFChkReader.get_next_block_params" class="docs-object-method">&nbsp;</a>
```python
get_next_block_params(self): 
```
Pulls the tag of the next block, the type, the number of bytes it'll be,
        and if it's a single-line block it'll also spit back the block itself
- `:returns`: `dict`
    >No description...

<a id="McUtils.GaussianInterface.GaussianImporter.GaussianFChkReader.get_block" class="docs-object-method">&nbsp;</a>
```python
get_block(self, name=None, dtype=None, byte_count=None, value=None): 
```
Pulls the next block by first pulling the block tag
- `:returns`: `_`
    >No description...

<a id="McUtils.GaussianInterface.GaussianImporter.GaussianFChkReader.skip_block" class="docs-object-method">&nbsp;</a>
```python
skip_block(self, name=None, dtype=None, byte_count=None, value=None): 
```
Skips the next block
- `:returns`: `_`
    >No description...

<a id="McUtils.GaussianInterface.GaussianImporter.GaussianFChkReader.parse" class="docs-object-method">&nbsp;</a>
```python
parse(self, keys=None, default='raise'): 
```

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/GaussianInterface/GaussianImporter/GaussianFChkReader.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/GaussianInterface/GaussianImporter/GaussianFChkReader.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/GaussianInterface/GaussianImporter/GaussianFChkReader.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/GaussianInterface/GaussianImporter/GaussianFChkReader.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/GaussianInterface/GaussianImporter.py?message=Update%20Docs)