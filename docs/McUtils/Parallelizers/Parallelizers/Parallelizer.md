## <a id="McUtils.Parallelizers.Parallelizers.Parallelizer">Parallelizer</a>
Abstract base class to help manage parallelism.
Provides the basic API that all parallelizers can be expected
to conform to.
Provides effectively the union of operations supported by
`mp.Pool` and `MPI`.
There is also the ability to lookup and register 'named'
parallelizers, since we expect a single program to not
really use more than one.
This falls back gracefully to the serial case.

### Properties and Methods
```python
default_printer: builtin_function_or_method
InMainProcess: type
InWorkerProcess: type
mode_map: dict
```
<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, logger=None, contract=None): 
```

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.load_registry" class="docs-object-method">&nbsp;</a>
```python
load_registry(): 
```

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.parallelizer_registry" class="docs-object-method">&nbsp;</a>
```python
@property
parallelizer_registry(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.get_default" class="docs-object-method">&nbsp;</a>
```python
get_default(): 
```
For compat.
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.lookup" class="docs-object-method">&nbsp;</a>
```python
lookup(key): 
```
Checks in the registry to see if a given parallelizer is there
        otherwise returns a `SerialNonParallelizer`.
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.register" class="docs-object-method">&nbsp;</a>
```python
register(self, key): 
```
Checks in the registry to see if a given parallelizer is there
        otherwise returns a `SerialNonParallelizer`.
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.active" class="docs-object-method">&nbsp;</a>
```python
@property
active(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.initialize" class="docs-object-method">&nbsp;</a>
```python
initialize(self): 
```
Initializes a parallelizer
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.finalize" class="docs-object-method">&nbsp;</a>
```python
finalize(self, exc_type, exc_val, exc_tb): 
```
Finalizes a parallelizer (if necessary)
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__enter__" class="docs-object-method">&nbsp;</a>
```python
__enter__(self): 
```
Allows the parallelizer context to be set
        using `with`
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__exit__" class="docs-object-method">&nbsp;</a>
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```
Allows the parallelizer context to be unset
        using `with`
- `exc_type`: `Any`
    >No description...
- `exc_val`: `Any`
    >No description...
- `exc_tb`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.on_main" class="docs-object-method">&nbsp;</a>
```python
@property
on_main(self): 
```
Returns whether or not the executing process is the main
        process or not
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.main_restricted" class="docs-object-method">&nbsp;</a>
```python
main_restricted(func): 
```
A decorator to indicate that a function should only be
        run when on the main process
- `func`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.worker_restricted" class="docs-object-method">&nbsp;</a>
```python
worker_restricted(func): 
```
A decorator to indicate that a function should only be
        run when on a worker process
- `func`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.send" class="docs-object-method">&nbsp;</a>
```python
send(self, data, loc, **kwargs): 
```
Sends data to the process specified by loc
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.receive" class="docs-object-method">&nbsp;</a>
```python
receive(self, data, loc, **kwargs): 
```
Receives data from the process specified by loc
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.broadcast" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.scatter" class="docs-object-method">&nbsp;</a>
```python
scatter(self, data, **kwargs): 
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

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.gather" class="docs-object-method">&nbsp;</a>
```python
gather(self, data, **kwargs): 
```
Performs a gather of data from the different
        available parallelizer processes
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.map" class="docs-object-method">&nbsp;</a>
```python
map(self, function, data, extra_args=None, extra_kwargs=None, **kwargs): 
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

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.starmap" class="docs-object-method">&nbsp;</a>
```python
starmap(self, function, data, extra_args=None, extra_kwargs=None, **kwargs): 
```
Performs a parallel map with unpacking of function over
        the held data on different processes
- `function`: `Any`
    >No description...
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, func, *args, main_kwargs=None, **kwargs): 
```
Runs the callable `func` in parallel
- `func`: `Any`
    >No description...
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.run" class="docs-object-method">&nbsp;</a>
```python
run(self, func, *args, comm=None, main_kwargs=None, **kwargs): 
```
Calls `apply`, but makes sure state is handled cleanly
- `func`: `Any`
    >No description...
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.from_config" class="docs-object-method">&nbsp;</a>
```python
from_config(mode=None, **kwargs): 
```

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.nprocs" class="docs-object-method">&nbsp;</a>
```python
@property
nprocs(self): 
```
Returns the number of processes the parallelizer has
        to work with
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.get_nprocs" class="docs-object-method">&nbsp;</a>
```python
get_nprocs(self): 
```
Returns the number of processes
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.id" class="docs-object-method">&nbsp;</a>
```python
@property
id(self): 
```
Returns some form of identifier for the current process
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.get_id" class="docs-object-method">&nbsp;</a>
```python
get_id(self): 
```
Returns the id for the current process
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.printer" class="docs-object-method">&nbsp;</a>
```python
@property
printer(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.main_print" class="docs-object-method">&nbsp;</a>
```python
main_print(self, *args, **kwargs): 
```
Prints from the main process
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.worker_print" class="docs-object-method">&nbsp;</a>
```python
worker_print(self, *args, **kwargs): 
```
Prints from a main worker process
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.print" class="docs-object-method">&nbsp;</a>
```python
print(self, *args, where='both', **kwargs): 
```
An implementation of print that operates differently on workers than on main
        processes
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.wait" class="docs-object-method">&nbsp;</a>
```python
wait(self): 
```
Causes all processes to wait until they've met up at this point.
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__repr__" class="docs-object-method">&nbsp;</a>
```python
__repr__(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.share" class="docs-object-method">&nbsp;</a>
```python
share(self, obj): 
```
Converts `obj` into a form that can be cleanly used with shared memory via a `SharedObjectManager`
- `obj`: `Any`
    >No description...
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parallelizers/Parallelizers/Parallelizer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parallelizers/Parallelizers/Parallelizer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parallelizers/Parallelizers/Parallelizer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parallelizers/Parallelizers/Parallelizer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/Parallelizers.py?message=Update%20Docs)