# <a id="McUtils.Parallelizers">McUtils.Parallelizers</a>
    
Provides utilities for setting up platform-independent parallelism
in a hopefully unobtrusive way.

This is used more extensively in `Psience`, but the design is to unify the MPI and `multiprocessing` APIs
so that one can simply pass in a `Parallelizer` object to a function and obtain parallelism over as
many processes as that object supports.
As a fallthrough, a `SerialNonParallelizer` is provided as a subclass that handles serial evaluation with
the same API so fewer special cases need to be checked.
Any function that supports parallelism should take the `parallelizer` keyword, which will be fed
the `Parallelizer` object itself.

### Members

  - [Parallelizer](Parallelizers/Parallelizers/Parallelizer.md)
  - [MultiprocessingParallelizer](Parallelizers/Parallelizers/MultiprocessingParallelizer.md)
  - [MPIParallelizer](Parallelizers/Parallelizers/MPIParallelizer.md)
  - [SerialNonParallelizer](Parallelizers/Parallelizers/SerialNonParallelizer.md)
  - [SendRecieveParallelizer](Parallelizers/Parallelizers/SendRecieveParallelizer.md)
  - [ClientServerRunner](Parallelizers/Runner/ClientServerRunner.md)

### Examples

The simplest parallelism is just parallelizing with `multiprocessing` over a single function

<div class="card in-out-block" markdown="1">

```python
def run_job(parallelizer=None):
    if parallelizer.on_main:
        data = np.arange(1000)
    else:
        data = None
    if parallelizer.on_main:
        flag = "woop"
    else:
        flag = None
    test = parallelizer.broadcast(flag) # send a flag from the main process to all the workers
    data = parallelizer.scatter(data)
    lens = parallelizer.gather(len(data))
    return lens

MultiprocessingParallelizer().run(run_job)
```
<div class="card-body out-block" markdown="1">

```python
[67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 66, 66, 66, 66, 66]
```

</div>
</div>

This will make sure a `Pool` of workers gets set up and will create communication channels from the main process to the works, then each process will run `run_job`, spreading the data out with `scatter` and bringing it back with `gather`.

This paradigm can be handled more simply with `map`. 
Here we'll map a function over blocks of data

<div class="card in-out-block" markdown="1">

```python
def mapped_func(self, data):
    return 1 + data
def map_applier(n=10, parallelizer=None):
    if parallelizer.on_main:
        data = np.arange(n)
    else:
        data = None
    return parallelizer.map(mapped_func, data)

MultiprocessingParallelizer().run(map_applier)
```

<div class="card-body out-block" markdown="1">

```python
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

</div>
</div>

but all of these will work equivalently well if the `parallelizer` were a `MPIParallelizer` instead (with correct run setup).

This also adapts itself well to more object-oriented solutions. 
Here's a sample class that can effectively use a parallelizer

```python
class SampleProgram:
    
    def __init__(self, nvals=1000, parallelizer=None):
        if not isinstance(parallelizer, Parallelizer):
            parallelizer = Parallelizer.lookup(parallelizer) # `Parallelizer` supports a registry in case you want to give a name
        self.par = parallelizer
        self.nvals = nvals
    
    def initialize_data(self):
        data = np.random.rand(self.nvals)
        # could be more expensive too
        return data
    
    def eval_parallel(self, data, parallelizer=None):
        data = parallelizer.scatter(data)
        # this would usually be much more sophisticated
        res = data**2
        return parallelizer.gather(res)
     
    @Parallelizer.main_restricted
    def run_main(self, parallelizer=None):
        """
        A function to be run by the main processes, setting
        up data, scattering, gathering, and post-processing
        """
        data = self.initialize_data()
        vals = self.eval_parallel(data, parallelizer=parallelizer)
        post_process = np.sqrt(vals)
        return post_process
        
    @Parallelizer.worker_restricted
    def run_worker(self, parallelizer=None):
        """
        A function to be run by the worker processes, really
        just doing the parallel work
        """
        self.eval_parallel(None, parallelizer=parallelizer)
    
    def run_par(self, parallelizer=None):
        """
        Something to be called by all processes
        """
        self.run_worker(parallelizer=parallelizer)
        return self.run_main(parallelizer=parallelizer)
    
    def run(self):
        """
        Boilerplate runner
        """
        print("Running with {}".format(self.par))
        return self.par.run(self.run_par)
```

and we can easily add in a `parallelizer` at run time.

First serial evaluation

<div class="card in-out-block" markdown="1">
```python
SampleProgram(nvals=10).run()
```
<div class="card-body out-block" markdown="1">

```lang-none
Running with SerialNonParallelizer(id=0, nprocs=1)

array([0.08772434, 0.18266685, 0.11234067, 0.4918653 , 0.30925003,
       0.43065691, 0.8271145 , 0.52147149, 0.13801914, 0.92917295])
```
</div>
</div>

but adding in parallelism is straightforward


<div class="card in-out-block" markdown="1">

```python
SampleProgram(nvals=10, parallelizer=MultiprocessingParallelizer()).run()
```

<div class="card-body out-block" markdown="1">

```lang-none
Running with MultiprocessingParallelizer(id=None, nprocs=None)

array([0.5852531 , 0.63836097, 0.40315219, 0.04769397, 0.5226616 ,
       0.68647924, 0.30869102, 0.01006922, 0.07439768, 0.83100183])
```

</div>
</div>

To support MPI-style calling, a `ClientServerRunner` is also provided.

### Unit Tests

```python
from Peeves.TestUtils import *
from McUtils.Parallelizers import *
from unittest import TestCase
import numpy as np, io, os, sys, tempfile as tmpf

# @Parallelizer.main_restricted
# def main_print(*args, parallelizer=None):
#     print(*args)
# @Parallelizer.worker_restricted
# def worker_print(*args, parallelizer=None):
#     print(*args)
# def run_job(parallelizer=None):
#     if parallelizer.on_main:
#         data = np.arange(1000)
#     else:
#         data = None
#     data = parallelizer.scatter(data)
#     lens = parallelizer.gather(len(data))
#     return lens

class ParallelizerTests(TestCase):

    # we don't really even need to send or get any state for these tests
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
    @debugTest
    def test_BasicMultiprocessing(self):
        par_lens = MultiprocessingParallelizer().run(self.run_job)
        serial_lens = SerialNonParallelizer().run(self.run_job)
        self.assertEquals(sum(par_lens), serial_lens)

    def mapped_func(self, data):
        return 1 + data
    def map_applier(self, n=1000, parallelizer=None):
        if parallelizer.on_main:
            data = np.arange(n)
        else:
            data = None
        return parallelizer.map(self.mapped_func, data)
    @validationTest
    def test_MapMultiprocessing(self):
        par_lens = MultiprocessingParallelizer().run(self.map_applier)
        serial_lens = SerialNonParallelizer().run(self.map_applier)
        self.assertEquals(par_lens, serial_lens)
    @validationTest
    def test_MapMultiprocessingDataSmall(self):
        par_lens = MultiprocessingParallelizer().run(self.map_applier, n=3, comm=[0, 1, 2])
        self.assertEquals(len(par_lens), 3)
        serial_lens = SerialNonParallelizer().run(self.map_applier, n=3)
        self.assertEquals(par_lens, serial_lens)

    def bcast_parallelizer(self, parallelizer=None):
        root_par = parallelizer.broadcast(parallelizer)
    @validationTest
    def test_BroadcastParallelizer(self):
        with MultiprocessingParallelizer() as parallelizer:
            parallelizer.run(self.bcast_parallelizer)
            parallelizer.run(self.bcast_parallelizer)

    def scatter_gather(self, n=1000, parallelizer=None):
        if parallelizer.on_main:
            data = np.arange(n)
        else:
            data = None
        data = parallelizer.scatter(data)
        l = len(data)
        res = parallelizer.gather(l)
        return res
    @validationTest
    def test_ScatterGatherMultiprocessing(self):
        p = MultiprocessingParallelizer()
        par_lens = p.run(self.scatter_gather)
        self.assertEquals(len(par_lens), p.nprocs+1)
        serial_lens = SerialNonParallelizer().run(self.scatter_gather)
        self.assertEquals(sum(par_lens), serial_lens)
    @validationTest
    def test_ScatterGatherMultiprocessingDataSmall(self):
        par_lens = MultiprocessingParallelizer().run(self.scatter_gather, 3, comm=[0, 1, 2])
        self.assertEquals(len(par_lens), 3)
        serial_lens = SerialNonParallelizer().run(self.scatter_gather, 3)
        self.assertEquals(sum(par_lens), serial_lens)

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
    @validationTest
    def test_MiscProblems(self):

        l = MultiprocessingParallelizer().run(self.simple_scatter_1, comm=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        MultiprocessingParallelizer().run(self.simple_print, comm=[0, 1, 2])
        # raise Exception(l)

```

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Parallelizers.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Parallelizers.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Parallelizers.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Parallelizers.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Parallelizers/__init__.py?message=Update%20Docs)