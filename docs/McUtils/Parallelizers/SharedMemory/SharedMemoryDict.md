## <a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict">SharedMemoryDict</a>
Implements a shared dict that uses
a managed dict to synchronize array metainfo
across processes

### Properties and Methods
<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, *seq, sync_dict=None, manager=None, marshaller=None, allocator=None, parallelizer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L548)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L548?message=Update%20Docs)]
</div>


- `marshaller`: `Any`
    >No description...
- `sync_dict`: `Any`
    >No description...
- `allocator`: `Any`
    >No description...
- `parallelizer`: `Any`
    >No description...

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.__getstate__" class="docs-object-method">&nbsp;</a> 
```python
__getstate__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L573)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L573?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.__contains__" class="docs-object-method">&nbsp;</a> 
```python
__contains__(self, item): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L578)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L578?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.__iter__" class="docs-object-method">&nbsp;</a> 
```python
__iter__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L580)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L580?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.__len__" class="docs-object-method">&nbsp;</a> 
```python
__len__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L582)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L582?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.__del__" class="docs-object-method">&nbsp;</a> 
```python
__del__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L584)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L584?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.keys" class="docs-object-method">&nbsp;</a> 
```python
keys(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L592)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L592?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.values" class="docs-object-method">&nbsp;</a> 
```python
values(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L594)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L594?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.items" class="docs-object-method">&nbsp;</a> 
```python
items(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L596)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L596?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.unshare" class="docs-object-method">&nbsp;</a> 
```python
unshare(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L598)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L598?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedMemoryDict.update" class="docs-object-method">&nbsp;</a> 
```python
update(self, v): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L601)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L601?message=Update%20Docs)]
</div>





___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Parallelizers/SharedMemory/SharedMemoryDict.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Parallelizers/SharedMemory/SharedMemoryDict.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Parallelizers/SharedMemory/SharedMemoryDict.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Parallelizers/SharedMemory/SharedMemoryDict.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/SharedMemory.py?message=Update%20Docs)