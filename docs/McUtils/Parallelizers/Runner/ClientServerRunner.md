## <a id="McUtils.Parallelizers.Runner.ClientServerRunner">ClientServerRunner</a>
Provides a framework for running MPI-like scripts in a client/server
model

### Properties and Methods
<a id="McUtils.Parallelizers.Runner.ClientServerRunner.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, client_runner: Callable, server_runner: Callable, parallelizer: McUtils.Parallelizers.Parallelizers.Parallelizer): 
```

<a id="McUtils.Parallelizers.Runner.ClientServerRunner.run" class="docs-object-method">&nbsp;</a>
```python
run(self): 
```
Runs the client/server processes depending on if the parallelizer
        is on the main or server processes
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parallelizers/Runner/ClientServerRunner.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parallelizers/Runner/ClientServerRunner.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parallelizers/Runner/ClientServerRunner.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parallelizers/Runner/ClientServerRunner.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/Runner.py?message=Update%20Docs)