## <a id="McUtils.Scaffolding.Caches.ObjectRegistry">ObjectRegistry</a>
Provides a simple interface to global object registries
so that pieces of code don't need to pass things like loggers
or parallelizers through every step of the code

### Properties and Methods
<a id="McUtils.Scaffolding.Caches.ObjectRegistry.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, default='raise'): 
```

<a id="McUtils.Scaffolding.Caches.ObjectRegistry.temp_default" class="docs-object-method">&nbsp;</a>
```python
temp_default(self, val): 
```

<a id="McUtils.Scaffolding.Caches.ObjectRegistry.lookup" class="docs-object-method">&nbsp;</a>
```python
lookup(self, key): 
```

<a id="McUtils.Scaffolding.Caches.ObjectRegistry.register" class="docs-object-method">&nbsp;</a>
```python
register(self, key, val): 
```

### Examples


