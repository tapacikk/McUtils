## <a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList">SharedMemoryList</a>
Implements a shared dict that uses
a managed dict to synchronize array metainfo
across processes

### Properties and Methods
<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *seq, sync_list=None, manager=None, marshaller=None, allocator=None, parallelizer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L483)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L483?message=Update%20Docs)]
</div>


- `marshaller`: `Any`
    >No description...
- `sync_dict`: `Any`
    >No description...
- `allocator`: `Any`
    >No description...
- `parallelizer`: `Any`
    >No description...

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__getstate__" class="docs-object-method">&nbsp;</a> 
```python
__getstate__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L508)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L508?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__contains__" class="docs-object-method">&nbsp;</a> 
```python
__contains__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L513)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L513?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__iter__" class="docs-object-method">&nbsp;</a> 
```python
__iter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L515)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L515?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__len__" class="docs-object-method">&nbsp;</a> 
```python
__len__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L517)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L517?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.__del__" class="docs-object-method">&nbsp;</a> 
```python
__del__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L519)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L519?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.unshare" class="docs-object-method">&nbsp;</a> 
```python
unshare(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L523)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L523?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.pop" class="docs-object-method">&nbsp;</a> 
```python
pop(self, k=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L526)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L526?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.insert" class="docs-object-method">&nbsp;</a> 
```python
insert(self, k, v): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L529)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L529?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.append" class="docs-object-method">&nbsp;</a> 
```python
append(self, v): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L532)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L532?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryList.extend" class="docs-object-method">&nbsp;</a> 
```python
extend(self, v): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L535)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L535?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Parallelizers/SharedMemory/SharedMemoryList.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/SharedMemory.py?message=Update%20Docs)