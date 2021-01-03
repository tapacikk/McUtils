## <a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer">Parallelizer</a>
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
parallelizer_registry: WeakValueDictionary
lookup: method
get_default: method
InMainProcess: type
InWorkerProcess: type
mode_map: dict
from_config: method
```
<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.register" class="docs-object-method">&nbsp;</a>
```python
register(self, key): 
```
Checks in the registry to see if a given parallelizer is there
        otherwise returns a `SerialNonParallelizer`.
- `key`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.set_default" class="docs-object-method">&nbsp;</a>
```python
set_default(self): 
```
Sets the parallelizer as the default one
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.reset_default" class="docs-object-method">&nbsp;</a>
```python
reset_default(self): 
```
Resets the default parallelizer
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.initialize" class="docs-object-method">&nbsp;</a>
```python
initialize(self): 
```
Initializes a parallelizer
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.finalize" class="docs-object-method">&nbsp;</a>
```python
finalize(self, exc_type, exc_val, exc_tb): 
```
Finalizes a parallelizer (if necessary)
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.__enter__" class="docs-object-method">&nbsp;</a>
```python
__enter__(self): 
```
Allows the parallelizer context to be set
        using `with`
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.__exit__" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.on_main" class="docs-object-method">&nbsp;</a>
```python
@property
on_main(self): 
```
Returns whether or not the executing process is the main
        process or not
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.main_restricted" class="docs-object-method">&nbsp;</a>
```python
main_restricted(func): 
```
A decorator to indicate that a function should only be
        run when on the main process
- `func`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.worker_restricted" class="docs-object-method">&nbsp;</a>
```python
worker_restricted(func): 
```
A decorator to indicate that a function should only be
        run when on a worker process
- `func`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.send" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.receive" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.broadcast" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.scatter" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.gather" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.map" class="docs-object-method">&nbsp;</a>
```python
map(self, function, data, **kwargs): 
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, func, *args, **kwargs): 
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

<a id="McUtils.McUtils.Parallelizers.Parallelizers.Parallelizer.run" class="docs-object-method">&nbsp;</a>
```python
run(self, func, *args, **kwargs): 
```

### Examples


