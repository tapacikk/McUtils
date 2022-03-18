## <a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager">FileBackedObjectManager</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/ObjectBackers.py#L74)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/ObjectBackers.py#L74?message=Update%20Docs)]
</div>

Provides an interface to back an object with
a serializer

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_directory: PersistenceLocation
```
<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, obj, chk=None, loc=None, checkpoint_class=<class 'McUtils.Scaffolding.Checkpointing.NumPyCheckpointer'>): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/ObjectBackers.py#L81)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/ObjectBackers.py#L81?message=Update%20Docs)]
</div>


- `obj`: `object`
    >the object to back
- `chk`: `Checkpointer`
    >a checkpointer to manage storing attributes
- `loc`: `str`
    >the location where attributes should be stored
- `checkpoint_class`: `Type[Checkpointer]`
    >a subclass of Checkpointer that implements the actual writing to disk

<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.basename" class="docs-object-method">&nbsp;</a> 
```python
@property
basename(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/ObjectBackers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/ObjectBackers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.get_basename" class="docs-object-method">&nbsp;</a> 
```python
get_basename(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/ObjectBackers.py#L117)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/ObjectBackers.py#L117?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.save_attr" class="docs-object-method">&nbsp;</a> 
```python
save_attr(self, attr): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/ObjectBackers.py#L124)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/ObjectBackers.py#L124?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.load_attr" class="docs-object-method">&nbsp;</a> 
```python
load_attr(self, attr): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Scaffolding/ObjectBackers.py#L129)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/ObjectBackers.py#L129?message=Update%20Docs)]
</div>

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Scaffolding/ObjectBackers/FileBackedObjectManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Scaffolding/ObjectBackers/FileBackedObjectManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Scaffolding/ObjectBackers/FileBackedObjectManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Scaffolding/ObjectBackers/FileBackedObjectManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Scaffolding/ObjectBackers.py#L74?message=Update%20Docs)