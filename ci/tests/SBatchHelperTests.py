
from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Misc import *
import sys, os, numpy as np

class SBatchHelperTests(TestCase):
    """
    Simple tests to make sure SBatch formatting is clean
    """

    @debugTest
    def test_BasicSBatch(self):
        sbatch = SBatchJob(
            """
            A simple job to check if 
            my SBatch server stuff is working the way it should
            """,
            job_name="myjob",
            mem='120GB',
            steps=[
                "echo 'yay'",
                "sleep 25s",
                "echo 'yayayayyasd'"
            ]
        ).format()

        print(sbatch)
