## <a id="McUtils.Misc.TemplateWriter.TemplateWriter">TemplateWriter</a>
A general class that can take a directory layout and apply template parameters to it
Very unsophisticated but workable

### Properties and Methods
```python
ignored_files: list
```
<a id="McUtils.Misc.TemplateWriter.TemplateWriter.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, template_dir, replacements=None, file_filter=None, **opts): 
```

<a id="McUtils.Misc.TemplateWriter.TemplateWriter.replacements" class="docs-object-method">&nbsp;</a>
```python
@property
replacements(self): 
```

<a id="McUtils.Misc.TemplateWriter.TemplateWriter.apply_replacements" class="docs-object-method">&nbsp;</a>
```python
apply_replacements(self, string): 
```
Applies the defined replacements to the
- `string`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.TemplateWriter.TemplateWriter.write_file" class="docs-object-method">&nbsp;</a>
```python
write_file(self, template_file, out_dir, apply_template=True, template_dir=None): 
```
writes a single _file_ to _dir_ and fills the template from the parameters passed when intializing the class
- `template_file`: `str`
    >the file to load and write into
- `out_dir`: `str`
    >the directory to write the file into
- `apply_template`: `bool`
    >whether to apply the template parameters to the file content or not
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.TemplateWriter.TemplateWriter.iterate_write" class="docs-object-method">&nbsp;</a>
```python
iterate_write(self, out_dir, apply_template=True, src_dir=None, template_dir=None): 
```
Iterates through the files in the template_dir and writes them out to dir
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Misc/TemplateWriter/TemplateWriter.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Misc/TemplateWriter/TemplateWriter.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Misc/TemplateWriter/TemplateWriter.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Misc/TemplateWriter/TemplateWriter.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Misc/TemplateWriter.py?message=Update%20Docs)