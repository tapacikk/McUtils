## <a id="McUtils.McUtils.Scaffolding.Jobs.Job">Job</a>
A job object to support simplified run scripting.
Provides a `job_data` checkpoint file that stores basic
data about job runtime and stuff, as well as a `logger` that
makes it easy to plug into a run time that supports logging

### Properties and Methods
```python
default_job_file: str
default_log_file: str
from_config: method
```
<a id="McUtils.McUtils.Scaffolding.Jobs.Job.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, job_dir, job_file=None, log_file=None, job_parameters=None): 
```

<a id="McUtils.McUtils.Scaffolding.Jobs.Job.path" class="docs-object-method">&nbsp;</a>
```python
path(self, *parts): 
```

- `parts`: `str`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Scaffolding.Jobs.Job.__enter__" class="docs-object-method">&nbsp;</a>
```python
__enter__(self): 
```

<a id="McUtils.McUtils.Scaffolding.Jobs.Job.__exit__" class="docs-object-method">&nbsp;</a>
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```

### Examples
