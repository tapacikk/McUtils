## <a id="McUtils.Extensions.ModuleLoader.ModuleLoader">ModuleLoader</a>
Provides a way to load dynamic modules.
Either use a `DynamicModuleLoader` or the `importlib.import_module` function
depending on how much customization is needed.

### Properties and Methods
<a id="McUtils.Extensions.ModuleLoader.ModuleLoader.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, rootdir='', rootpkg=None, retag=False): 
```

- `rootdir`: `str`
    >root directory to look for files off of
- `rootpkg`: `str or None`
    >root package to look for files off of

<a id="McUtils.Extensions.ModuleLoader.ModuleLoader.load" class="docs-object-method">&nbsp;</a>
```python
load(self, file, pkg=None): 
```

### Examples


