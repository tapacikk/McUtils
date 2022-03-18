## <a id="McUtils.Parallelizers.Parallelizers.Parallelizer">Parallelizer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L43)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L43?message=Update%20Docs)]
</div>

Abstract base class to help manage parallelism.
Provides the basic API that all parallelizers can be expected
to conform to.
Provides effectively the union of operations supported by
`mp.Pool` and `MPI`.
There is also the ability to lookup and register 'named'
parallelizers, since we expect a single program to not
really use more than one.
This falls back gracefully to the serial case.

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
default_printer: builtin_function_or_method
InMainProcess: type
InWorkerProcess: type
mode_map: dict
```
<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, logger=None, contract=None, uid=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L63)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L63?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.load_registry" class="docs-object-method">&nbsp;</a> 
```python
load_registry(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L79)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L79?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.parallelizer_registry" class="docs-object-method">&nbsp;</a> 
```python
@property
parallelizer_registry(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.get_default" class="docs-object-method">&nbsp;</a> 
```python
get_default(): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L88)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L88?message=Update%20Docs)]
</div>

For compat.
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.lookup" class="docs-object-method">&nbsp;</a> 
```python
lookup(key): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L97)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L97?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L109)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L109?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.initialize" class="docs-object-method">&nbsp;</a> 
```python
initialize(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L124)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L124?message=Update%20Docs)]
</div>

Initializes a parallelizer
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.finalize" class="docs-object-method">&nbsp;</a> 
```python
finalize(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L133)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L133?message=Update%20Docs)]
</div>

Finalizes a parallelizer (if necessary)
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__enter__" class="docs-object-method">&nbsp;</a> 
```python
__enter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L143)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L143?message=Update%20Docs)]
</div>

Allows the parallelizer context to be set
        using `with`
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__exit__" class="docs-object-method">&nbsp;</a> 
```python
__exit__(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L159)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L159?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

Returns whether or not the executing process is the main
        process or not
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.main_restricted" class="docs-object-method">&nbsp;</a> 
```python
main_restricted(func): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L205)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L205?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L227)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L227?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L257)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L257?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L270)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L270?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L283)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L283?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L296)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L296?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L313)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L313?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L332)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L332?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L349)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L349?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L367)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L367?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L382)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L382?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L400)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L400?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.nprocs" class="docs-object-method">&nbsp;</a> 
```python
@property
nprocs(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

Returns the number of processes the parallelizer has
        to work with
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.get_nprocs" class="docs-object-method">&nbsp;</a> 
```python
get_nprocs(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L425)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L425?message=Update%20Docs)]
</div>

Returns the number of processes
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.id" class="docs-object-method">&nbsp;</a> 
```python
@property
id(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

Returns some form of identifier for the current process
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.pid" class="docs-object-method">&nbsp;</a> 
```python
@property
pid(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.get_id" class="docs-object-method">&nbsp;</a> 
```python
get_id(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L446)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L446?message=Update%20Docs)]
</div>

Returns the id for the current process
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.printer" class="docs-object-method">&nbsp;</a> 
```python
@property
printer(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.main_print" class="docs-object-method">&nbsp;</a> 
```python
main_print(self, *args, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L464)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L464?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L475)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L475?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L486)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L486?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L505)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L505?message=Update%20Docs)]
</div>

Causes all processes to wait until they've met up at this point.
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.__repr__" class="docs-object-method">&nbsp;</a> 
```python
__repr__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L514)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L514?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.Parallelizer.share" class="docs-object-method">&nbsp;</a> 
```python
share(self, obj): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L525)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L525?message=Update%20Docs)]
</div>

Converts `obj` into a form that can be cleanly used with shared memory via a `SharedObjectManager`
- `obj`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Parallelizers/Parallelizers/Parallelizer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Parallelizers/Parallelizers/Parallelizer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Parallelizers/Parallelizers/Parallelizer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Parallelizers/Parallelizers/Parallelizer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L43?message=Update%20Docs)