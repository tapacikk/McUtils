## <a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer">MultiprocessingParallelizer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L814)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L814?message=Update%20Docs)]
</div>

Parallelizes using a  process pool and a runner
function that represents a "main loop".

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

```python
SendRecvQueuePair: type
PoolCommunicator: ABCMeta
```
<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, worker=False, pool: <bound method BaseContext.Pool of <multiprocessing.context.DefaultContext instance>> = None, context=None, manager=None, logger=None, contract=None, comm=None, rank=None, allow_restart=True, initialization_timeout=0.5, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L977)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L977?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.get_nprocs" class="docs-object-method">&nbsp;</a> 
```python
get_nprocs(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1002)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1002?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.get_id" class="docs-object-method">&nbsp;</a> 
```python
get_id(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1004)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1004?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.comm" class="docs-object-method">&nbsp;</a> 
```python
@property
comm(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

Returns the communicator used by the paralellizer
- `:returns`: `MultiprocessingParallelizer.PoolCommunicator`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.__getstate__" class="docs-object-method">&nbsp;</a> 
```python
__getstate__(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1038)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1038?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.__setstate__" class="docs-object-method">&nbsp;</a> 
```python
__setstate__(self, state): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1053)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1053?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.apply" class="docs-object-method">&nbsp;</a> 
```python
apply(self, func, *args, comm=None, main_kwargs=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1119)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1119?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1205)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1205?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.get_pool_nprocs" class="docs-object-method">&nbsp;</a> 
```python
get_pool_nprocs(pool): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1208)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1208?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.initialize" class="docs-object-method">&nbsp;</a> 
```python
initialize(self, allow_restart=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1223)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1223?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.finalize" class="docs-object-method">&nbsp;</a> 
```python
finalize(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1251)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1251?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.on_main" class="docs-object-method">&nbsp;</a> 
```python
@property
on_main(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.MultiprocessingParallelizer.from_config" class="docs-object-method">&nbsp;</a> 
```python
from_config(**kw): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Parallelizers/Parallelizers.py#L1261)/[edit](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L1261?message=Update%20Docs)]
</div>

 </div>
</div>



<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
### <a class="collapse-link" data-toggle="collapse" href="#tests">Tests</a> <a class="float-right" data-toggle="collapse" href="#tests"><i class="fa fa-chevron-down"></i></a>
 </div>
<div class="collapsible-section collapsible-section-body collapse show" id="tests" markdown="1">

- [BasicMultiprocessing](#BasicMultiprocessing)
- [MapMultiprocessing](#MapMultiprocessing)
- [MapMultiprocessingDataSmall](#MapMultiprocessingDataSmall)
- [ScatterGatherMultiprocessing](#ScatterGatherMultiprocessing)
- [ScatterGatherMultiprocessingDataSmall](#ScatterGatherMultiprocessingDataSmall)
- [MiscProblems](#MiscProblems)
- [DistributedDict](#DistributedDict)

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

#### <a name="BasicMultiprocessing">BasicMultiprocessing</a>
```python
    def test_BasicMultiprocessing(self):
        par_lens = MultiprocessingParallelizer().run(self.run_job)
        serial_lens = SerialNonParallelizer().run(self.run_job)
        self.assertEquals(sum(par_lens), serial_lens)
```
#### <a name="MapMultiprocessing">MapMultiprocessing</a>
```python
    def test_MapMultiprocessing(self):
        par_lens = MultiprocessingParallelizer().run(self.map_applier)
        serial_lens = SerialNonParallelizer().run(self.map_applier)
        self.assertEquals(par_lens, serial_lens)
```
#### <a name="MapMultiprocessingDataSmall">MapMultiprocessingDataSmall</a>
```python
    def test_MapMultiprocessingDataSmall(self):
        par_lens = MultiprocessingParallelizer().run(self.map_applier, n=3, comm=[0, 1, 2])
        self.assertEquals(len(par_lens), 3)
        serial_lens = SerialNonParallelizer().run(self.map_applier, n=3)
        self.assertEquals(par_lens, serial_lens)
```
#### <a name="ScatterGatherMultiprocessing">ScatterGatherMultiprocessing</a>
```python
    def test_ScatterGatherMultiprocessing(self):
        p = MultiprocessingParallelizer()
        par_lens = p.run(self.scatter_gather)
        self.assertEquals(len(par_lens), p.nprocs+1)
        serial_lens = SerialNonParallelizer().run(self.scatter_gather)
        self.assertEquals(sum(par_lens), serial_lens)
```
#### <a name="ScatterGatherMultiprocessingDataSmall">ScatterGatherMultiprocessingDataSmall</a>
```python
    def test_ScatterGatherMultiprocessingDataSmall(self):
        par_lens = MultiprocessingParallelizer().run(self.scatter_gather, 3, comm=[0, 1, 2])
        self.assertEquals(len(par_lens), 3)
        serial_lens = SerialNonParallelizer().run(self.scatter_gather, 3)
        self.assertEquals(sum(par_lens), serial_lens)
```
#### <a name="MiscProblems">MiscProblems</a>
```python
    def test_MiscProblems(self):

        l = MultiprocessingParallelizer().run(self.simple_scatter_1, comm=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        MultiprocessingParallelizer().run(self.simple_print, comm=[0, 1, 2])
```
#### <a name="DistributedDict">DistributedDict</a>
```python
    def test_DistributedDict(self):

         my_data = {'a':np.random.rand(10, 5, 5), 'b':np.random.rand(10, 3, 8), 'c':np.random.rand(10, 15, 4), 'd':{}}

         par = MultiprocessingParallelizer(processes=2, logger=Logger())
         my_data = par.share(my_data)

         par.run(self.mutate_shared_dict, my_data)

         self.assertEquals(my_data['a'][1, 0, 0], 5.0)

         my_data = my_data.unshare()
         self.assertIsInstance(my_data, dict)
         self.assertIsInstance(my_data['a'], np.ndarray)
```

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Parallelizers/Parallelizers/MultiprocessingParallelizer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Parallelizers/Parallelizers/MultiprocessingParallelizer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Parallelizers/Parallelizers/MultiprocessingParallelizer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Parallelizers/Parallelizers/MultiprocessingParallelizer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/master/Parallelizers/Parallelizers.py#L814?message=Update%20Docs)