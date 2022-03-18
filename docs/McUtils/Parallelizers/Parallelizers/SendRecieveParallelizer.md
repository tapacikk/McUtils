## <a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer">SendRecieveParallelizer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L545)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L545?message=Update%20Docs)]
</div>

Parallelizer that implements `scatter`, `gather`, `broadcast`, and `map`
based on just having a communicator that supports `send` and `receive methods

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
ReceivedError: type
SendReceieveCommunicator: ABCMeta
```
<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.comm" class="docs-object-method">&nbsp;</a> 
```python
@property
comm(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

Returns the communicator used by the paralellizer
- `:returns`: `SendRecieveParallelizer.SendReceieveCommunicator`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.send" class="docs-object-method">&nbsp;</a> 
```python
send(self, data, loc, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L610)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L610?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L624)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L624?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L642)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L642?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L661)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L661?message=Update%20Docs)]
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.gather" class="docs-object-method">&nbsp;</a> 
```python
gather(self, data, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L696)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L696?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L733)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L733?message=Update%20Docs)]
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.starmap" class="docs-object-method">&nbsp;</a> 
```python
starmap(self, func, data, extra_args=None, extra_kwargs=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L771)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L771?message=Update%20Docs)]
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

<a id="McUtils.Parallelizers.Parallelizers.SendRecieveParallelizer.wait" class="docs-object-method">&nbsp;</a> 
```python
wait(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L803)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L803?message=Update%20Docs)]
</div>

Causes all processes to wait until they've met up at this point.
- `:returns`: `_`
    >No description...

 </div>
</div>




___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Parallelizers/Parallelizers/SendRecieveParallelizer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L545?message=Update%20Docs)