## <a id="McUtils.McUtils.GaussianInterface.GaussianJob.GaussianJob">GaussianJob</a>
A class that writes Gaussian .gjf files given a system and config/template options

### Properties and Methods
```python
job_template_dir: str
Job: type
Config: type
System: type
```
<a id="McUtils.McUtils.GaussianInterface.GaussianJob.GaussianJob.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, name, *args, description=None, system=None, job=None, config=None, template='Template.gjf', file=None): 
```

<a id="McUtils.McUtils.GaussianInterface.GaussianJob.GaussianJob.format" class="docs-object-method">&nbsp;</a>
```python
format(self): 
```

<a id="McUtils.McUtils.GaussianInterface.GaussianJob.GaussianJob.write" class="docs-object-method">&nbsp;</a>
```python
write(self, file=None): 
```

<a id="McUtils.McUtils.GaussianInterface.GaussianJob.GaussianJob.start" class="docs-object-method">&nbsp;</a>
```python
start(self, *cmd, binary='g09', **kwargs): 
```
Starts a Gaussian job
- `cmd`: `Any`
    >No description...
- `binary`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >started process

<a id="McUtils.McUtils.GaussianInterface.GaussianJob.GaussianJob.run" class="docs-object-method">&nbsp;</a>
```python
run(self, *args, **kwargs): 
```

<a id="McUtils.McUtils.GaussianInterface.GaussianJob.GaussianJob.__str__" class="docs-object-method">&nbsp;</a>
```python
__str__(self): 
```

### Examples
