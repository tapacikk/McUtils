## <a id="McUtils.Scaffolding.Configurations.Config">Config</a>
A configuration object which basically just supports
a dictionary interface, but which also can automatically
filter itself so that it only provides the keywords supported
by a `from_config` method.

### Properties and Methods
```python
config_file_name: str
config_file_extensions: list
find_config: method
get_serializer: method
new: method
```
<a id="McUtils.Scaffolding.Configurations.Config.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, config, serializer=None): 
```
Loads the config from a file
- `config`: `str`
    >No description...
- `serializer`: `None | BaseSerializer`
    >No description...

<a id="McUtils.Scaffolding.Configurations.Config.serialize" class="docs-object-method">&nbsp;</a>
```python
serialize(self, file, ops): 
```

<a id="McUtils.Scaffolding.Configurations.Config.deserialize" class="docs-object-method">&nbsp;</a>
```python
deserialize(self, file): 
```

<a id="McUtils.Scaffolding.Configurations.Config.save" class="docs-object-method">&nbsp;</a>
```python
save(self): 
```

<a id="McUtils.Scaffolding.Configurations.Config.load" class="docs-object-method">&nbsp;</a>
```python
load(self): 
```

<a id="McUtils.Scaffolding.Configurations.Config.name" class="docs-object-method">&nbsp;</a>
```python
@property
name(self): 
```

<a id="McUtils.Scaffolding.Configurations.Config.opt_dict" class="docs-object-method">&nbsp;</a>
```python
@property
opt_dict(self): 
```

<a id="McUtils.Scaffolding.Configurations.Config.filter" class="docs-object-method">&nbsp;</a>
```python
filter(self, keys, strict=True): 
```
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
Applies func to stored parameters
- `func`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Configurations.Config.update" class="docs-object-method">&nbsp;</a>
```python
update(self, **kw): 
```

<a id="McUtils.Scaffolding.Configurations.Config.load_opts" class="docs-object-method">&nbsp;</a>
```python
load_opts(self): 
```

<a id="McUtils.Scaffolding.Configurations.Config.get_conf_attr" class="docs-object-method">&nbsp;</a>
```python
get_conf_attr(self, item): 
```

<a id="McUtils.Scaffolding.Configurations.Config.__getattr__" class="docs-object-method">&nbsp;</a>
```python
__getattr__(self, item): 
```

### Examples


