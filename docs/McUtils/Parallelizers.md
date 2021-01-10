# <a id="McUtils.Parallelizers">McUtils.Parallelizers</a>
    
Provides utilities for setting up platform-independent parallelism
in a hopefully unobtrusive way

### Members:

  - [Parallelizer](Parallelizers/Parallelizers/Parallelizer.md)
  - [MultiprocessingParallelizer](Parallelizers/Parallelizers/MultiprocessingParallelizer.md)
  - [MPIParallelizer](Parallelizers/Parallelizers/MPIParallelizer.md)
  - [SerialNonParallelizer](Parallelizers/Parallelizers/SerialNonParallelizer.md)

### Examples:



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
        # self.main_print("Go!")
        # self.worker_print("...")
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

    def mapped_func(self, id):
        return 1 + id
    def map_applier(self, parallelizer=None):
        if parallelizer.on_main:
            data = np.arange(1000)
        else:
            data = None
        return parallelizer.map(self.mapped_func, data)
    @debugTest
    def test_MapMultiprocessing(self):
        par_lens = MultiprocessingParallelizer().run(self.map_applier)
        serial_lens = SerialNonParallelizer().run(self.map_applier)
        self.assertEquals(par_lens, serial_lens)

    def bcast_parallelizer(self, parallelizer=None):
        root_par = parallelizer.broadcast(parallelizer)
    @debugTest
    def test_BroadcastParallelizer(self):
        with MultiprocessingParallelizer() as parallelizer:
            parallelizer.run(self.bcast_parallelizer)
            parallelizer.run(self.bcast_parallelizer)

```