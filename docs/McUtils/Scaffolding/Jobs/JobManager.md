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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Jobs.py#L179)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Jobs.py#L179?message=Update%20Docs)]
</div>

<a id="McUtils.Scaffolding.Jobs.JobManager.job" class="docs-object-method">&nbsp;</a> 
```python
job(self, name, timestamp=False, **kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Jobs.py#L184)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Jobs.py#L184?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Jobs.py#L204)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Jobs.py#L204?message=Update%20Docs)]
</div>

A special case convenience function that goes
        directly to starting a job from a folder
- `:returns`: `Job`
    >No description...

<a id="McUtils.Scaffolding.Jobs.JobManager.current_job" class="docs-object-method">&nbsp;</a> 
```python
current_job(job_type=None, make_config=True, **opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Scaffolding/Jobs.py#L223)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Scaffolding/Jobs.py#L223?message=Update%20Docs)]
</div>

A special case convenience function that starts a
        JobManager one directory up from the current
        working directory and intializes a job from the
        current working directory
- `:returns`: `Job`
    >No description...




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [Jobbing](#Jobbing)
- [JobInit](#JobInit)
- [CurrentJob](#CurrentJob)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
#### <a class="collapse-link" data-toggle="collapse" href="#test-setup">Setup</a> <a class="float-right" data-toggle="collapse" href="#test-setup"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="test-setup" markdown="1">

Before we can run our examples we should get a bit of setup out of the way.
Since these examples were harvested from the unit tests not all pieces
will be necessary for all situations.
```python
from Peeves.TestUtils import *
from McUtils.Scaffolding import *
import McUtils.Parsers as parsers
from unittest import TestCase
import numpy as np, io, os, sys, tempfile as tmpf
```

All tests are wrapped in a test class
```python
class ScaffoldingTests(TestCase):
    class DataHolderClass:
        def __init__(self, **keys):
            self.data = keys
        def to_state(self, serializer=None):
            return self.data
        @classmethod
        def from_state(cls, state, serializer=None):
            return cls(**state)
```

 </div>
</div>

#### <a name="Jobbing">Jobbing</a>
```python
    def test_Jobbing(self):

        import time

        with tmpf.TemporaryDirectory() as temp_dir:

            manager = JobManager(temp_dir)
            with manager.job("test") as job:
                logger = job.logger
                with logger.block(tag="Sleeping"):
                    logger.log_print("Goodnight!")
                    time.sleep(.2)
                    logger.log_print("Okee I'm back up")

            self.assertEquals(os.path.basename(job.dir), "test")
            self.assertEquals(set(job.checkpoint.backend.keys()), {'start', 'runtime'})
            with open(job.logger.log_file) as doopy:
                doop_str = doopy.read()
                self.assertNotEqual("", doop_str)
```
#### <a name="JobInit">JobInit</a>
```python
    def test_JobInit(self):

        import time

        with tmpf.TemporaryDirectory() as temp_dir:
            manager = JobManager(temp_dir)
            with manager.job(TestManager.test_data("persistence_tests/test_job")) as job:
                logger = job.logger

                with logger.block(tag="Sleeping"):
                    logger.log_print("Goodnight!")
                    time.sleep(.2)
                    logger.log_print("Okee I'm back up")

            self.assertEquals(os.path.basename(job.dir), "test_job")
            self.assertEquals(set(Config(job.dir).opt_dict.keys()), {'logger', 'parallelizer', 'config_location'})
            self.assertEquals(set(job.checkpoint.backend.keys()), {'start', 'runtime'})
            with open(job.logger.log_file) as doopy:
                doop_str = doopy.read()
                self.assertNotEqual("", doop_str)
```
#### <a name="CurrentJob">CurrentJob</a>
```python
    def test_CurrentJob(self):

        import time

        with tmpf.TemporaryDirectory() as temp_dir:
            jobby = JobManager.job_from_folder(temp_dir)
            with jobby as job:
                logger = job.logger

                with logger.block(tag="Sleeping"):
                    logger.log_print("Goodnight!")
                    time.sleep(.2)
                    logger.log_print("Okee I'm back up")

            with open(job.logger.log_file) as doopy:
                doop_str = doopy.read()
                self.assertNotEqual("", doop_str)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Scaffolding/Jobs/JobManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Scaffolding/Jobs/JobManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Scaffolding/Jobs/JobManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Scaffolding/Jobs/JobManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Scaffolding/Jobs.py?message=Update%20Docs)