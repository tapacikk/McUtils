## <a id="McUtils.Scaffolding.Persistence.PersistenceManager">PersistenceManager</a>
Defines a manager that can load configuration data from a directory
or, maybe in the future, a SQL database or similar.
Requires class that supports `from_config` to load and `to_config` to save.

### Properties and Methods
<a id="McUtils.Scaffolding.Persistence.PersistenceManager.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, cls, persistence_loc=None): 
```

- `cls`: `type`
    >No description...
- `persistence_loc`: `str | None`
    >location from which to load/save objects

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.obj_loc" class="docs-object-method">&nbsp;</a>
```python
obj_loc(self, key): 
```

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.load_config" class="docs-object-method">&nbsp;</a>
```python
load_config(self, key, make_new=False, init=None): 
```
Loads the config for the persistent structure named `key`
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.new_config" class="docs-object-method">&nbsp;</a>
```python
new_config(self, key, init=None): 
```
Creates a new space and config for the persistent structure named `key`
- `key`: `str`
    >name for job
- `init`: `str | dict | None`
    >initial parameters
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.contains" class="docs-object-method">&nbsp;</a>
```python
contains(self, key): 
```
Checks if `key` is a supported persistent structure
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.load" class="docs-object-method">&nbsp;</a>
```python
load(self, key, make_new=False, strict=True, init=None): 
```
Loads the persistent structure named `key`
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Persistence.PersistenceManager.save" class="docs-object-method">&nbsp;</a>
```python
save(self, obj): 
```
Saves requisite config data for a structure
- `obj`: `Any`
    >No description...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Persistence/PersistenceManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Persistence/PersistenceManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Persistence/PersistenceManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Persistence/PersistenceManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Persistence.py?message=Update%20Docs)