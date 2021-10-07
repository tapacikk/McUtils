## <a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer">MPIParallelizer</a>
Parallelizes using `mpi4py`

### Properties and Methods
```python
MPICommunicator: ABCMeta
```
<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, root=0, comm=None): 
```

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.get_nprocs" class="docs-object-method">&nbsp;</a>
```python
get_nprocs(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.get_id" class="docs-object-method">&nbsp;</a>
```python
get_id(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.initialize" class="docs-object-method">&nbsp;</a>
```python
initialize(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.finalize" class="docs-object-method">&nbsp;</a>
```python
finalize(self, exc_type, exc_val, exc_tb): 
```

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.comm" class="docs-object-method">&nbsp;</a>
```python
@property
comm(self): 
```
Returns the communicator used by the paralellizer
- `:returns`: `MPIParallelizer.MPICommunicator`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.on_main" class="docs-object-method">&nbsp;</a>
```python
@property
on_main(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.broadcast" class="docs-object-method">&nbsp;</a>
```python
broadcast(self, data, **kwargs): 
```
Sends the same data to all processes
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.scatter" class="docs-object-method">&nbsp;</a>
```python
scatter(self, data, shape=None, **kwargs): 
```
Performs a scatter of data to the different
        available parallelizer processes.
        *NOTE:* unlike in the MPI case, `data` does not
        need to be evenly divisible by the number of available
        processes
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.gather" class="docs-object-method">&nbsp;</a>
```python
gather(self, data, shape=None, **kwargs): 
```
Performs a gather of data from the different
        available parallelizer processes
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.map" class="docs-object-method">&nbsp;</a>
```python
map(self, func, data, input_shape=None, output_shape=None, **kwargs): 
```
Performs a parallel map of function over
        the held data on different processes
- `function`: `Any`
    >No description...
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, func, *args, **kwargs): 
```
Applies func to args in parallel on all of the processes.
        For MPI, since jobs are always started with mpirun, this
        is just a regular apply
- `func`: `Any`
    >No description...
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MPIParallelizer.from_config" class="docs-object-method">&nbsp;</a>
```python
from_config(**kw): 
```

### Examples




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parallelizers/Parallelizers/MPIParallelizer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parallelizers/Parallelizers/MPIParallelizer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parallelizers/Parallelizers/MPIParallelizer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parallelizers/Parallelizers/MPIParallelizer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/Parallelizers.py?message=Update%20Docs)