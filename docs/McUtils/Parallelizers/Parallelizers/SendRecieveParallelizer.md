## <a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer">SendRecieveParallelizer</a>
Parallelizer that implements `scatter`, `gather`, `broadcast`, and `map`
based on just having a communicator that supports `send` and `receive methods

### Properties and Methods
```python
SendReceieveCommunicator: ABCMeta
```
<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.comm" class="docs-object-method">&nbsp;</a>
```python
@property
comm(self): 
```
Returns the communicator used by the paralellizer
- `:returns`: `SendRecieveParallelizer.SendReceieveCommunicator`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.send" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.receive" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.broadcast" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.scatter" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.gather" class="docs-object-method">&nbsp;</a>
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.map" class="docs-object-method">&nbsp;</a>
```python
map(self, func, data, extra_args=None, extra_kwargs=None, **kwargs): 
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.starmap" class="docs-object-method">&nbsp;</a>
```python
starmap(self, func, data, extra_args=None, extra_kwargs=None, **kwargs): 
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.wait" class="docs-object-method">&nbsp;</a>
```python
wait(self): 
```
Causes all processes to wait until they've met up at this point.
- `:returns`: `_`
    >No description...





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/Parallelizers.py?message=Update%20Docs)