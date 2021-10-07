## <a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager">FileBackedObjectManager</a>
Provides an interface to back an object with
a serializer

### Properties and Methods
```python
default_directory: PersistenceLocation
```
<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, obj, chk=None, loc=None, checkpoint_class=<class 'McUtils.Scaffolding.Checkpointing.NumPyCheckpointer'>): 
```

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

<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.get_basename" class="docs-object-method">&nbsp;</a>
```python
get_basename(self): 
```

<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.save_attr" class="docs-object-method">&nbsp;</a>
```python
save_attr(self, attr): 
```

<a id="McUtils.Scaffolding.ObjectBackers.FileBackedObjectManager.load_attr" class="docs-object-method">&nbsp;</a>
```python
load_attr(self, attr): 
```

### Examples


