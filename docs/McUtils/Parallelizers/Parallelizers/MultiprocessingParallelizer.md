## <a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer">MultiprocessingParallelizer</a>
Parallelizes using a  process pool and a runner
function that represents a "main loop".

### Properties and Methods
```python
SendRecvQueuePair: type
PoolCommunicator: ABCMeta
from_config: method
```
<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.__init__" class="docs-object-method">&nbsp;</a>
```python
__init__(self, worker=False, pool: <bound method BaseContext.Pool of <multiprocessing.context.DefaultContext instance>> = None, context=None, manager=None, printer=None, verbose=False, initialization_timeout=0.5, **kwargs): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.get_nprocs" class="docs-object-method">&nbsp;</a>
```python
get_nprocs(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.get_id" class="docs-object-method">&nbsp;</a>
```python
get_id(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.comm" class="docs-object-method">&nbsp;</a>
```python
@property
comm(self): 
```
Returns the communicator used by the paralellizer
- `:returns`: `MultiprocessingParallelizer.PoolCommunicator`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.__getstate__" class="docs-object-method">&nbsp;</a>
```python
__getstate__(self): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.apply" class="docs-object-method">&nbsp;</a>
```python
apply(self, func, *args, **kwargs): 
```
Applies func to args in parallel on all of the processes
- `func`: `Any`
    >No description...
- `args`: `Any`
    >No description...
- `kwargs`: `Any`
    >No description...
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.get_pool_context" class="docs-object-method">&nbsp;</a>
```python
get_pool_context(pool): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.get_pool_nprocs" class="docs-object-method">&nbsp;</a>
```python
get_pool_nprocs(pool): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.initialize" class="docs-object-method">&nbsp;</a>
```python
initialize(self, allow_restart=True): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.finalize" class="docs-object-method">&nbsp;</a>
```python
finalize(self, exc_type, exc_val, exc_tb): 
```

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.on_main" class="docs-object-method">&nbsp;</a>
```python
@property
on_main(self): 
```

### Examples


