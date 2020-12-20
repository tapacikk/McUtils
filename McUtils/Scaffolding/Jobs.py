"""
A job management package to make it easier to instantiate
job
"""

import time, datetime, os
from .Persistence import PersistenceManager
from .Checkpointing import JSONCheckpointer
from .Logging import Logger

__all__ = [
    "Job",
    "JobManager"
]

class Job:
    """
    A job object to support simplified run scripting.
    Provides a `job_data` checkpoint file that stores basic
    data about job runtime and stuff, as well as a `logger` that
    makes it easy to plug into a run time that supports logging
    """
    default_job_file = "job_data.json"
    default_log_file = "log.txt"
    def __init__(self,
                 job_dir,
                 job_file=None,
                 log_file=None,
                 job_parameters=None
                 ):
        self.dir = job_dir
        self.end = None
        if job_file is None:
            job_file = self.default_job_file
        if os.path.abspath(job_file) != job_file:
            job_file = os.path.join(job_dir, job_file)
        self.checkpoint = JSONCheckpointer(job_file)

        if log_file is None:
            log_file = self.default_log_file
        if os.path.abspath(log_file) != log_file:
            log_file = os.path.join(job_dir, log_file)
        self.logger = Logger(log_file)

        self._init_dir = None

        self.job_parameters = job_parameters

    @classmethod
    def from_config(cls,
                    location=None,
                    job_file=None,
                    log_file=None,
                    job_parameters=None
                    ):
        return cls(location,
                   job_file=job_file,
                   log_file=log_file,
                   job_parameters=job_parameters
                   )

    def path(self, *parts):
        """
        :param parts:
        :type parts: str
        :return:
        :rtype:
        """
        return os.path.join(self.dir, *parts)

    def __enter__(self):
        self._init_dir = os.getcwd()
        os.chdir(self.dir)
        self.checkpoint.__enter__()
        self.start = time.time()
        self.checkpoint['start'] = {
            'datetime': datetime.datetime.now().isoformat(),
            'timestamp': self.start
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._init_dir = os.getcwd()
        os.chdir(self.dir)
        end = time.time()
        self.checkpoint['runtime'] = end - self.start
        self.checkpoint.__enter__()

class JobManager(PersistenceManager):
    """
    A class to manage job instances.
    Thin layer on a `PersistenceManager`
    """
    def __init__(self, job_dir, job_type=Job):
        super().__init__(job_type, persistence_loc=job_dir)

    def job(self, name, timestamp=False, **kw):
        """
        Returns a loaded or new job with the given name and settings
        :param name:
        :type name: str
        :param timestamp:
        :type timestamp:
        :param kw:
        :type kw:
        :return:
        :rtype:
        """
        if timestamp:
            name += "_" + datetime.datetime.now().isoformat()
        return self.load(name, make_new=True, init=kw)