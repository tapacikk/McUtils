## <a id="McUtils.Scaffolding.Configurations.Config">Config</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L12)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L12?message=Update%20Docs)]
</div>

A configuration object which basically just supports
a dictionary interface, but which also can automatically
filter itself so that it only provides the keywords supported
by a `from_config` method.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
config_file_name: str
config_file_extensions: list
```
<a id="McUtils.Scaffolding.Configurations.Config.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, config, serializer=None, extra_params=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L19)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L19?message=Update%20Docs)]
</div>

Loads the config from a file
- `config`: `str`
    >No description...
- `serializer`: `None | BaseSerializer`
    >No description...

<a id="McUtils.Scaffolding.Configurations.Config.find_config" class="docs-object-method">&nbsp;</a> 
```python
find_config(config, name=None, extensions=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L36)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L36?message=Update%20Docs)]
</div>

Finds configuration file (if config isn't a file)
- `config`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Configurations.Config.get_serializer" class="docs-object-method">&nbsp;</a> 
```python
get_serializer(file): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L64)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L64?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.new" class="docs-object-method">&nbsp;</a> 
```python
new(loc, init=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L72)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L72?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.serialize" class="docs-object-method">&nbsp;</a> 
```python
serialize(self, file, ops): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L84)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L84?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.deserialize" class="docs-object-method">&nbsp;</a> 
```python
deserialize(self, file): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L91)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L91?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.save" class="docs-object-method">&nbsp;</a> 
```python
save(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L99)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L99?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.load" class="docs-object-method">&nbsp;</a> 
```python
load(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L101)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L101?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.name" class="docs-object-method">&nbsp;</a> 
```python
@property
name(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.opt_dict" class="docs-object-method">&nbsp;</a> 
```python
@property
opt_dict(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.filter" class="docs-object-method">&nbsp;</a> 
```python
filter(self, keys, strict=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L116)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L116?message=Update%20Docs)]
</div>

Returns a filtered option dictionary according to keys.
        Strict mode will raise an error if there is a key in the config that isn't
        in keys.
- `keys`: `Iterable[str] | function`
    >No description...
- `strict`: `bool`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Configurations.Config.apply" class="docs-object-method">&nbsp;</a> 
```python
apply(self, func, strict=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L151)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L151?message=Update%20Docs)]
</div>

Applies func to stored parameters
- `func`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Configurations.Config.update" class="docs-object-method">&nbsp;</a> 
```python
update(self, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L163)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L163?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.load_opts" class="docs-object-method">&nbsp;</a> 
```python
load_opts(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L168)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L168?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.get_conf_attr" class="docs-object-method">&nbsp;</a> 
```python
get_conf_attr(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L176)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L176?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Configurations.Config.__getattr__" class="docs-object-method">&nbsp;</a> 
```python
__getattr__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Configurations.py#L184)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L184?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/Configurations/Config.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/Configurations/Config.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/Configurations/Config.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/Configurations/Config.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Configurations.py#L12?message=Update%20Docs)