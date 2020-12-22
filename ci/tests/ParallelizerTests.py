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
    data = np.arange(1000)
    data = parallelizer.scatter(data)
    # main_print(data, len(data))
    # worker_print(data[0], data[-1])
    lens = parallelizer.gather(len(data))
    main_print(lens)

class ParallelizerTests(TestCase):

    def test_BasicMultiprocessing(self):

        MultiprocessingParallelizer().run(
            run_job
        )



