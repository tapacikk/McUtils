## <a id="McUtils.Scaffolding.Jobs.JobManager">JobManager</a>
A class to manage job instances.
Thin layer on a `PersistenceManager`

### Properties and Methods
```python
default_job_type: type
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
- `:returns`: `Job`
    >No description...

<a id="McUtils.Scaffolding.Jobs.JobManager.job_from_folder" class="docs-object-method">&nbsp;</a>
```python
job_from_folder(folder, job_type=None, make_config=True, **opts): 
```
A special case convenience function that goes
        directly to starting a job from a folder
- `:returns`: `Job`
    >No description...

<a id="McUtils.Scaffolding.Jobs.JobManager.current_job" class="docs-object-method">&nbsp;</a>
```python
current_job(job_type=None, make_config=True, **opts): 
```
A special case convenience function that starts a
        JobManager one directory up from the current
        working directory and intializes a job from the
        current working directory
- `:returns`: `Job`
    >No description...

### Examples


