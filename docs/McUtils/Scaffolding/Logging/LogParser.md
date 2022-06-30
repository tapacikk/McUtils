## <a id="McUtils.Scaffolding.Logging.LogParser">LogParser</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L374)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L374?message=Update%20Docs)]
</div>

A parser that will take a log file and stream it as a series of blocks

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
LogBlockParser: type
```
<a id="McUtils.Scaffolding.Logging.LogParser.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, file, block_settings=None, block_level_padding=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L378)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L378?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.LogParser.get_block_settings" class="docs-object-method">&nbsp;</a> 
```python
get_block_settings(self, block_level): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L387)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L387?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.LogParser.get_block" class="docs-object-method">&nbsp;</a> 
```python
get_block(self, level=0, tag=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L516)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L516?message=Update%20Docs)]
</div>


- `level`: `Any`
    >No description...
- `tag`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Logging.LogParser.get_line" class="docs-object-method">&nbsp;</a> 
```python
get_line(self, level=0, tag=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L551)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L551?message=Update%20Docs)]
</div>


- `level`: `Any`
    >No description...
- `tag`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Logging.LogParser.get_blocks" class="docs-object-method">&nbsp;</a> 
```python
get_blocks(self, tag=None, level=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L573)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L573?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Logging.LogParser.get_lines" class="docs-object-method">&nbsp;</a> 
```python
get_lines(self, tag=None, level=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/Logging.py#L587)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L587?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Logging/LogParser.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Logging/LogParser.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Logging/LogParser.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Logging/LogParser.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/Logging.py#L374?message=Update%20Docs)