## <a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer">SerialNonParallelizer</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1685)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1685?message=Update%20Docs)]
</div>

Totally serial evaluation for cases where no parallelism
is provide

<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
 
### <a class="collapse-link" data-toggle="collapse" href="#methods">Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>

 </div>
 <div class="collapsible-section collapsible-section-body collapse" id="methods" markdown="1">

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.get_nprocs" class="docs-object-method">&nbsp;</a> 
```python
get_nprocs(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1691)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1691?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.get_id" class="docs-object-method">&nbsp;</a> 
```python
get_id(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1693)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1693?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.initialize" class="docs-object-method">&nbsp;</a> 
```python
initialize(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1696)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1696?message=Update%20Docs)]
</div>

Initializes a parallelizer
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.finalize" class="docs-object-method">&nbsp;</a> 
```python
finalize(self, exc_type, exc_val, exc_tb): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1704)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1704?message=Update%20Docs)]
</div>

Finalizes a parallelizer (if necessary)
        if necessary
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.on_main" class="docs-object-method">&nbsp;</a> 
```python
@property
on_main(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L?message=Update%20Docs)]
</div>

Returns whether or not the executing process is the main
        process or not
- `:returns`: `_`
    >No description...

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.send" class="docs-object-method">&nbsp;</a> 
```python
send(self, data, loc, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1721)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1721?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1733)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1733?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1745)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1745?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1757)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1757?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1769)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1769?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1782)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1782?message=Update%20Docs)]
</div>

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
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1806)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1806?message=Update%20Docs)]
</div>

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
apply(self, func, *args, comm=None, main_kwargs=None, **kwargs): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1829)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1829?message=Update%20Docs)]
</div>

<a id="McUtils.Parallelizers.Parallelizers.SerialNonParallelizer.wait" class="docs-object-method">&nbsp;</a> 
```python
wait(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/edit/Parallelizers/Parallelizers.py#L1835)/[edit](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1835?message=Update%20Docs)]
</div>

No need to wait when you're in a serial environment
- `:returns`: `_`
    >No description...

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

 </div>
</div>

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Parallelizers/Parallelizers/SerialNonParallelizer.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Parallelizers/Parallelizers/SerialNonParallelizer.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Parallelizers/Parallelizers/SerialNonParallelizer.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Parallelizers/Parallelizers/SerialNonParallelizer.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/Parallelizers/Parallelizers.py#L1685?message=Update%20Docs)