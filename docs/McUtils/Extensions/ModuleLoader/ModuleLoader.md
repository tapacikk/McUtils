## <a id="McUtils.Extensions.ModuleLoader.ModuleLoader">ModuleLoader</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ModuleLoader.py#L102)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ModuleLoader.py#L102?message=Update%20Docs)]
</div>

Provides a way to load dynamic modules.
Either use a `DynamicModuleLoader` or the `importlib.import_module` function
depending on how much customization is needed.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Extensions.ModuleLoader.ModuleLoader.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, rootdir='', rootpkg=None, retag=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ModuleLoader.py#L108)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ModuleLoader.py#L108?message=Update%20Docs)]
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
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Extensions/ModuleLoader.py#L123)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ModuleLoader.py#L123?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Extensions/ModuleLoader/ModuleLoader.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Extensions/ModuleLoader/ModuleLoader.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Extensions/ModuleLoader/ModuleLoader.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Extensions/ModuleLoader/ModuleLoader.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Extensions/ModuleLoader.py#L102?message=Update%20Docs)