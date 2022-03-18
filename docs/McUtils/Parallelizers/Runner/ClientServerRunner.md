## <a id="McUtils.Parallelizers.Runner.ClientServerRunner">ClientServerRunner</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Runner.py#L8)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Runner.py#L8?message=Update%20Docs)]
</div>

Provides a framework for running MPI-like scripts in a client/server
model

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Parallelizers.Runner.ClientServerRunner.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, client_runner: Callable, server_runner: Callable, parallelizer: McUtils.Parallelizers.Parallelizers.Parallelizer): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Runner.py#L14)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Runner.py#L14?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Runner.ClientServerRunner.run" class="docs-object-method">&nbsp;</a> 
```python
run(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Runner.py#L19)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Runner.py#L19?message=Update%20Docs)]
</div>

Runs the client/server processes depending on if the parallelizer
        is on the main or server processes
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Parallelizers/Runner/ClientServerRunner.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Parallelizers/Runner/ClientServerRunner.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Parallelizers/Runner/ClientServerRunner.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Parallelizers/Runner/ClientServerRunner.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Runner.py#L8?message=Update%20Docs)