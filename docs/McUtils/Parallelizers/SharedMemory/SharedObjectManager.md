## <a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager">SharedObjectManager</a>
Provides a high-level interface to create a manager
that supports shared memory objects through the multiprocessing
interface
Only supports data that can be marshalled into a NumPy array.

### Properties and Methods
```python
primitive_types: tuple
```
<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, obj, base_dict=None, parallelizer=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L624)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L624?message=Update%20Docs)]
</div>


- `mem_manager`: `Any`
    >a memory manager like `multiprocessing.SharedMemoryManager`
- `obj`: `Any`
    >the object whose attributes should be given by shared memory objects
- `base_dict`: `SharedMemoryDict`
    >the dict that stores the shared arrays (can also be shared)

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.is_primitive" class="docs-object-method">&nbsp;</a> 
```python
is_primitive(val): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L648)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L648?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.save_attr" class="docs-object-method">&nbsp;</a> 
```python
save_attr(self, attr): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L652)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L652?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.del_attr" class="docs-object-method">&nbsp;</a> 
```python
del_attr(self, attr): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L660)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L660?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.load_attr" class="docs-object-method">&nbsp;</a> 
```python
load_attr(self, attr): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L666)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L666?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.get_saved_keys" class="docs-object-method">&nbsp;</a> 
```python
get_saved_keys(self, obj): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L673)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L673?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.save_keys" class="docs-object-method">&nbsp;</a> 
```python
save_keys(self, keys=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L676)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L676?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.share" class="docs-object-method">&nbsp;</a> 
```python
share(self, keys=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L682)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L682?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.load_keys" class="docs-object-method">&nbsp;</a> 
```python
load_keys(self, keys=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L694)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L694?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.unshare" class="docs-object-method">&nbsp;</a> 
```python
unshare(self, keys=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L700)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L700?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.__del__" class="docs-object-method">&nbsp;</a> 
```python
__del__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L724)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L724?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.list" class="docs-object-method">&nbsp;</a> 
```python
list(self, *l): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L731)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L731?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.dict" class="docs-object-method">&nbsp;</a> 
```python
dict(self, *d): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L738)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L738?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.SharedMemory.SharedObjectManager.array" class="docs-object-method">&nbsp;</a> 
```python
array(self, a): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/SharedMemory.py#L745)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/SharedMemory.py#L745?message=Update%20Docs)]
</div>




<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [MakeSharedMem](#MakeSharedMem)

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
#### <a class="collapse-link" data-toggle="collapse" href="#test-setup">Setup</a> <a class="float-right" data-toggle="collapse" href="#test-setup"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="test-setup" markdown="1">

Before we can run our examples we should get a bit of setup out of the way.
Since these examples were harvested from the unit tests not all pieces
will be necessary for all situations.
```python
from Peeves.TestUtils import *
from McUtils.Scaffolding import Logger
from McUtils.Parallelizers import *
from unittest import TestCase
import numpy as np, io, os, sys, tempfile as tmpf
```

All tests are wrapped in a test class
```python
class ParallelizerTests(TestCase):
    def __getstate__(self):
        return {}
    def __setstate__(self, state):
        pass
    def run_job(self, parallelizer=None):
        if parallelizer.on_main:
            data = np.arange(1000)
        else:
            data = None
        if parallelizer.on_main:
            flag = "woop"
        else:
            flag = None
        test = parallelizer.broadcast(flag)
        # self.worker_print(test)
        data = parallelizer.scatter(data)
        lens = parallelizer.gather(len(data))
        return lens
    def mapped_func(self, data):
        return 1 + data
    def map_applier(self, n=1000, parallelizer=None):
        if parallelizer.on_main:
            data = np.arange(n)
        else:
            data = None
        return parallelizer.map(self.mapped_func, data)
    def bcast_parallelizer(self, parallelizer=None):
        root_par = parallelizer.broadcast(parallelizer)
    def scatter_gather(self, n=1000, parallelizer=None):
        if parallelizer.on_main:
            data = np.arange(n)
        else:
            data = None
        data = parallelizer.scatter(data)
        l = len(data)
        res = parallelizer.gather(l)
        return res
    def simple_scatter_1(self, parallelizer=None):
        data = [
            np.array([[0, 0]]), np.array([[0, 1]]), np.array([[0, 2]]),
            np.array([[1, 0]]), np.array([[1, 1]]), np.array([[1, 2]]),
            np.array([[2, 0]]), np.array([[2, 1]]), np.array([[2, 2]])
        ]
        data = parallelizer.scatter(data)
        l = len(data)
        l = parallelizer.gather(l)
        return l
    def simple_print(self, parallelizer=None):
        parallelizer.print(1)
    def mutate_shared_dict(self, d, parallelizer=None):
        wat = d['d']
        parallelizer.print('{a} {b} {c} {d}', a=id(wat), b=id(d['d']), c=id(d['d']), d=d)
        if not parallelizer.on_main:
            d['a'][1, 0, 0] = 5
            wat['key'] = 5
        parallelizer.print('{v} {g}', v=wat, g=d['d'])
```

 </div>
</div>

#### <a name="MakeSharedMem">MakeSharedMem</a>
```python
    def test_MakeSharedMem(self):

        a = np.random.rand(10, 5, 5)
        manager = SharedObjectManager(a)

        saved = manager.share()
        loaded = manager.unshare() #type: np.ndarray
        # print(type(loaded), loaded.shape, loaded.data, loaded.size)

        self.assertTrue(np.allclose(a, loaded))
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/McUtils/Parallelizers/SharedMemory/SharedObjectManager.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/SharedMemory.py?message=Update%20Docs)