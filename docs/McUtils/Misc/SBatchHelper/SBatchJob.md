## <a id="McUtils.Misc.SBatchHelper.SBatchJob">SBatchJob</a>
Provides a simple interface to formatting SLURM
files so that they can be submitted to `sbatch`.
The hope is that this can be subclassed codify
options for different HPC paritions and whatnot.

### Properties and Methods
```python
slurm_keys: list
default_opts: dict
sbatch_opt_template: str
sbatch_template: str
sbatch_enter_command: str
sbatch_exit_command: str
```
<a id="McUtils.Misc.SBatchHelper.SBatchJob.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, description=None, job_name=None, account=None, partition=None, mem=None, nodes=None, ntasks_per_node=None, chdir=None, output=None, steps=(), **opts): 
```

<a id="McUtils.Misc.SBatchHelper.SBatchJob.clean_opts" class="docs-object-method">&nbsp;</a>
```python
clean_opts(self, opts): 
```
Makes sure opt names are clean.
        Does no validation of the values sent in.
- `opts`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.SBatchHelper.SBatchJob.format_opt_block" class="docs-object-method">&nbsp;</a>
```python
format_opt_block(self): 
```
Formats block of options
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.SBatchHelper.SBatchJob.format" class="docs-object-method">&nbsp;</a>
```python
format(self): 
```
Formats an SBATCH file from the held options
- `call_steps`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

### Examples


