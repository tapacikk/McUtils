## <a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer">ModuleSerializer</a>
A somewhat hacky serializer that supports module-based serialization.
Writes all module parameters to a dict with a given attribute.
Serialization doesn't support loading arbitrary python code, but deserialization does.
Use at your own risk.

### Properties and Methods
```python
default_loader: NoneType
default_attr: str
```
<a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, attr=None, loader=None): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer.loader" class="docs-object-method">&nbsp;</a>
```python
@property
loader(self): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer.attr" class="docs-object-method">&nbsp;</a>
```python
@property
attr(self): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer.convert" class="docs-object-method">&nbsp;</a>
```python
convert(self, data): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer.deconvert" class="docs-object-method">&nbsp;</a>
```python
deconvert(self, data): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer.serialize" class="docs-object-method">&nbsp;</a>
```python
serialize(self, file, data, **kwargs): 
```

<a id="McUtils.McUtils.Scaffolding.Serializers.ModuleSerializer.deserialize" class="docs-object-method">&nbsp;</a>
```python
deserialize(self, file, key=None, **kwargs): 
```

### Examples


