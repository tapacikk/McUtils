## <a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer">SerialNonParallelizer</a>
Totally serial evaluation for cases where no parallelism
is provide

### Properties and Methods
<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.get_nprocs" class="docs-object-method">&nbsp;</a>
```python
get_nprocs(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.get_id" class="docs-object-method">&nbsp;</a>
```python
get_id(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.initialize" class="docs-object-method">&nbsp;</a>
```python
initialize(self): 
```
Initializes a parallelizer
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.finalize" class="docs-object-method">&nbsp;</a>
```python
finalize(self, exc_type, exc_val, exc_tb): 
```
Finalizes a parallelizer (if necessary)
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.on_main" class="docs-object-method">&nbsp;</a>
```python
@property
on_main(self): 
```
Returns whether or not the executing process is the main
        process or not
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.send" class="docs-object-method">&nbsp;</a>
```python
send(self, data, loc, **kwargs): 
```
A no-op
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.receive" class="docs-object-method">&nbsp;</a>
```python
receive(self, data, loc, **kwargs): 
```
A no-op
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.broadcast" class="docs-object-method">&nbsp;</a>
```python
broadcast(self, data, **kwargs): 
```
A no-op
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.scatter" class="docs-object-method">&nbsp;</a>
```python
scatter(self, data, **kwargs): 
```
A no-op
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.gather" class="docs-object-method">&nbsp;</a>
```python
gather(self, data, **kwargs): 
```
A no-op
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.map" class="docs-object-method">&nbsp;</a>
```python
map(self, function, data, extra_args=None, extra_kwargs=None, **kwargs): 
```
Performs a serial map of the function over
        the passed data
- `function`: `Any`
    >No description...
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.starmap" class="docs-object-method">&nbsp;</a>
```python
starmap(self, function, data, extra_args=None, extra_kwargs=None, **kwargs): 
```
Performs a serial map with unpacking of the function over
        the passed data
- `function`: `Any`
    >No description...
- `data`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, func, *args, **kwargs): 
```

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.wait" class="docs-object-method">&nbsp;</a>
```python
wait(self): 
```
No need to wait when you're in a serial environment
- `:returns`: `_`
    >No description...

### Examples


