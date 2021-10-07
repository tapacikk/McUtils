## <a id="McUtils.Scaffolding.Jobs.Job">Job</a>
A job object to support simplified run scripting.
Provides a `job_data` checkpoint file that stores basic
data about job runtime and stuff, as well as a `logger` that
makes it easy to plug into a run time that supports logging

### Properties and Methods
```python
default_job_file: str
default_log_file: str
```
<a id="McUtils.Scaffolding.Jobs.Job.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, job_dir, job_file=None, logger=None, parallelizer=None, job_parameters=None): 
```

<a id="McUtils.Scaffolding.Jobs.Job.from_config" class="docs-object-method">&nbsp;</a>
```python
from_config(config_location=None, job_file=None, logger=None, parallelizer=None, job_parameters=None): 
```

<a id="McUtils.Scaffolding.Jobs.Job.load_checkpoint" class="docs-object-method">&nbsp;</a>
```python
load_checkpoint(self, job_file): 
```
Loads the checkpoint we'll use to dump params
- `job_file`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Jobs.Job.load_logger" class="docs-object-method">&nbsp;</a>
```python
load_logger(self, log_spec): 
```
Loads the appropriate logger
- `log_spec`: `str | dict`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Jobs.Job.load_parallelizer" class="docs-object-method">&nbsp;</a>
```python
load_parallelizer(self, par_spec): 
```
Loads the appropriate parallelizer.
        If something other than a dict is passed,
        tries out multiple specs sequentially until it finds one that works
- `log_spec`: `dict`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Jobs.Job.path" class="docs-object-method">&nbsp;</a>
```python
path(self, *parts): 
```

- `parts`: `str`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Scaffolding.Jobs.Job.working_directory" class="docs-object-method">&nbsp;</a>
```python
@property
working_directory(self): 
```

<a id="McUtils.Scaffolding.Jobs.Job.__enter__" class="docs-object-method">&nbsp;</a>
```python
__enter__(self): 
```

<a id="McUtils.Scaffolding.Jobs.Job.__exit__" class="docs-object-method">&nbsp;</a>
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Scaffolding/Jobs/Job.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Scaffolding/Jobs/Job.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Scaffolding/Jobs/Job.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Scaffolding/Jobs/Job.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Jobs.py?message=Update%20Docs)