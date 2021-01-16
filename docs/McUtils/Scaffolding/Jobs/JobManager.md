## <a id="McUtils.Scaffolding.Jobs.JobManager">JobManager</a>
A class to manage job instances.
Thin layer on a `PersistenceManager`

### Properties and Methods
```python
default_job_type: type
job_from_folder: method
current_job: method
```
<a id="McUtils.Scaffolding.Jobs.JobManager.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, job_dir, job_type=None): 
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


