## <a id="McUtils.GaussianInterface.GaussianJob.GaussianJob">GaussianJob</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianJob.py#L78)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L78?message=Update%20Docs)]
</div>

A class that writes Gaussian .gjf files given a system and config/template options

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
job_template_dir: str
Job: type
Config: type
System: type
```
<a id="McUtils.GaussianInterface.GaussianJob.GaussianJob.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, name, *args, description=None, system=None, job=None, config=None, template='TemplateTerse.gjf', footer=None, file=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianJob.py#L83)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L83?message=Update%20Docs)]
</div>

<a id="McUtils.GaussianInterface.GaussianJob.GaussianJob.format" class="docs-object-method">&nbsp;</a> 
```python
format(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianJob.py#L135)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L135?message=Update%20Docs)]
</div>

Formats the job string
- `:returns`: `_`
    >No description...

<a id="McUtils.GaussianInterface.GaussianJob.GaussianJob.write" class="docs-object-method">&nbsp;</a> 
```python
write(self, file=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianJob.py#L156)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L156?message=Update%20Docs)]
</div>

Writes the job to a file
- `file`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.GaussianInterface.GaussianJob.GaussianJob.start" class="docs-object-method">&nbsp;</a> 
```python
start(self, *cmd, binary='g09', **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianJob.py#L179)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L179?message=Update%20Docs)]
</div>

Starts a Gaussian job
- `cmd`: `Any`
    >No description...
- `binary`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >started process

<a id="McUtils.GaussianInterface.GaussianJob.GaussianJob.run" class="docs-object-method">&nbsp;</a> 
```python
run(self, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianJob.py#L206)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L206?message=Update%20Docs)]
</div>

<a id="McUtils.GaussianInterface.GaussianJob.GaussianJob.__str__" class="docs-object-method">&nbsp;</a> 
```python
__str__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/GaussianInterface/GaussianJob.py#L210)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L210?message=Update%20Docs)]
</div>

 </div>
</div>

```python
from McUtils.GaussianInterface import GaussianJob 

job = GaussianJob(
        "water scan",
        description="Simple water scan",
        config= GaussianJob.Config(
            NProc = 4,
            Mem = '1000MB'
        ),
        job= GaussianJob.Job(
            'Scan'
        ),
        system = GaussianJob.System(
            charge=0,
            molecule=[
                ["O", "H", "H"],
                [
                    [0, 0, 0],
                    [.987, 0, 0],
                    [0, .987, 0]
                ]
            ],
            vars=[
                GaussianJob.System.Variable("y1", 0., 10., .1),
                GaussianJob.System.Constant("x1", 10)
            ]
    
        )
    )
# print(job.write(), file=sys.stderr)
self.assertIsInstance(job.format(), str)
```


___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/GaussianInterface/GaussianJob/GaussianJob.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/GaussianInterface/GaussianJob/GaussianJob.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/GaussianInterface/GaussianJob/GaussianJob.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/GaussianInterface/GaussianJob/GaussianJob.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/GaussianInterface/GaussianJob.py#L78?message=Update%20Docs)