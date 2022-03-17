## <a id="McUtils.Extensions.ModuleLoader.ModuleLoader">ModuleLoader</a>
Provides a way to load dynamic modules.
Either use a `DynamicModuleLoader` or the `importlib.import_module` function
depending on how much customization is needed.

### Properties and Methods
<a id="McUtils.Extensions.ModuleLoader.ModuleLoader.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, rootdir='', rootpkg=None, retag=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ModuleLoader.py#L108)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ModuleLoader.py#L108?message=Update%20Docs)]
</div>


- `rootdir`: `str`
    >root directory to look for files off of
- `rootpkg`: `str or None`
    >root package to look for files off of

<a id="McUtils.Extensions.ModuleLoader.ModuleLoader.load" class="docs-object-method">&nbsp;</a> 
```python
load(self, file, pkg=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Extensions/ModuleLoader.py#L123)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Extensions/ModuleLoader.py#L123?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Extensions/ModuleLoader/ModuleLoader.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Extensions/ModuleLoader/ModuleLoader.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Extensions/ModuleLoader/ModuleLoader.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Extensions/ModuleLoader/ModuleLoader.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Extensions/ModuleLoader.py?message=Update%20Docs)