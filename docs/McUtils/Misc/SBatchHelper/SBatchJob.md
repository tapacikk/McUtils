## <a id="McUtils.Misc.SBatchHelper.SBatchJob">SBatchJob</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Misc/SBatchHelper.py#L9)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Misc/SBatchHelper.py#L9?message=Update%20Docs)]
</div>

Provides a simple interface to formatting SLURM
files so that they can be submitted to `sbatch`.
The hope is that this can be subclassed codify
options for different HPC paritions and whatnot.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Misc/SBatchHelper.py#L38)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Misc/SBatchHelper.py#L38?message=Update%20Docs)]
</div>

<a id="McUtils.Misc.SBatchHelper.SBatchJob.clean_opts" class="docs-object-method">&nbsp;</a> 
```python
clean_opts(self, opts): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Misc/SBatchHelper.py#L60)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Misc/SBatchHelper.py#L60?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Misc/SBatchHelper.py#L80)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Misc/SBatchHelper.py#L80?message=Update%20Docs)]
</div>

Formats block of options
- `:returns`: `_`
    >No description...

<a id="McUtils.Misc.SBatchHelper.SBatchJob.format" class="docs-object-method">&nbsp;</a> 
```python
format(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Misc/SBatchHelper.py#L107)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Misc/SBatchHelper.py#L107?message=Update%20Docs)]
</div>

Formats an SBATCH file from the held options
- `call_steps`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Misc/SBatchHelper/SBatchJob.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Misc/SBatchHelper/SBatchJob.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Misc/SBatchHelper/SBatchJob.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Misc/SBatchHelper/SBatchJob.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Misc/SBatchHelper.py#L9?message=Update%20Docs)