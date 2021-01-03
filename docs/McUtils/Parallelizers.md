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


@Parallelizer.main_restricted
def main_print(*args, parallelizer=None):
    print(*args)
@Parallelizer.worker_restricted
def worker_print(*args, parallelizer=None):
    print(*args)
def run_job(parallelizer=None):
    if parallelizer.on_main:
        data = np.arange(1000)
    else:
        data = None
    data = parallelizer.scatter(data)
    lens = parallelizer.gather(len(data))
    return lens
class ParallelizerTests(TestCase):
    def test_BasicMultiprocessing(self):
        par_lens = MultiprocessingParallelizer().run(run_job)
        serial_lens = SerialNonParallelizer().run(run_job)
        self.assertEquals(sum(par_lens), serial_lens)




```