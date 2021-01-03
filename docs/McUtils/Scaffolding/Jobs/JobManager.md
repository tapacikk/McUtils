## <a id="McUtils.Scaffolding.Jobs.JobManager">JobManager</a>
A class to manage job instances.
Thin layer on a `PersistenceManager`

### Properties and Methods
<a id="McUtils.Scaffolding.Jobs.JobManager.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, job_dir, job_type=<class 'McUtils.Scaffolding.Jobs.Job'>): 
```

<a id="McUtils.Scaffolding.Jobs.JobManager.job" class="docs-object-method">&nbsp;</a>
```python
job(self, name, timestamp=False, **kw): 
```
Returns a loaded or new job with the given name and settings
- `name`: `str`
    >No description...
- `timestamp`: `Any`
    >No description...
- `kw`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples


