"""
A job management package to make it easier to instantiate
job
"""

import time, datetime, os, shutil
from .Persistence import PersistenceManager
from .Checkpointing import JSONCheckpointer
from .Logging import Logger

from ..Parallelizers import Parallelizer

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
                 logger=None,
                 parallelizer=None,
                 job_parameters=None
                 ):
        self.dir = job_dir
        self.end = None

        self.checkpoint = self.load_checkpoint(job_file)
        self.logger = self.load_logger(logger)
        self.parallelizer = self.load_parallelizer(parallelizer)

        self._init_dir = None

        self.job_parameters = job_parameters

    @classmethod
    def from_config(cls,
                    config_location=None,
                    job_file=None,
                    logger=None,
                    parallelizer=None,
                    job_parameters=None
                    ):
        return cls(config_location,
                   job_file=job_file,
                   logger=logger,
                   parallelizer=parallelizer,
                   job_parameters=job_parameters
                   )

    def load_checkpoint(self, job_file):
        """
        Loads the checkpoint we'll use to dump params

        :param job_file:
        :type job_file:
        :return:
        :rtype:
        """
        if job_file is None:
            job_file = self.default_job_file
        if os.path.abspath(job_file) != job_file:
            job_file = os.path.join(self.dir, job_file)
        return JSONCheckpointer(job_file)
    def load_logger(self, log_spec):
        """
        Loads the appropriate logger

        :param log_spec:
        :type log_spec: str | dict
        :return:
        :rtype:
        """
        if log_spec is None:
            log_spec = self.default_log_file
        if isinstance(log_spec, str):
            if os.path.abspath(log_spec) != log_spec:
                log_spec = os.path.join(self.dir, log_spec)
            return Logger(log_spec)
        elif isinstance(log_spec, dict):
            if 'log_file' not in log_spec:
                log_spec['log_file'] = self.default_log_file
            if os.path.abspath(log_spec['log_file']) != log_spec['log_file']:
                log_spec['log_file'] = os.path.join(self.dir, log_spec['log_file'])
            return Logger(**log_spec)
        else:
            raise ValueError("don't know what to do with log spec {}".format(
                log_spec
            ))

    def load_parallelizer(self, par_spec):
        """
        Loads the appropriate parallelizer.
        If something other than a dict is passed,
        tries out multiple specs sequentially until it finds one that works

        :param log_spec:
        :type log_spec: dict
        :return:
        :rtype:
        """
        if par_spec is None:
            return None
        elif isinstance(par_spec, dict):
            try:
                par = Parallelizer.from_config(**par_spec)
            except (ImportError, ModuleNotFoundError):
                return None
            else:
                return par
        else:
            for spec in par_spec:
                par = self.load_parallelizer(spec)
                if par is not None:
                    return par

    def path(self, *parts):
        """
        :param parts:
        :type parts: str
        :return:
        :rtype:
        """
        return os.path.join(self.dir, *parts)

    @property
    def working_directory(self):
        if 'working_directory' in self.job_parameters:
            init_dir = os.getcwd()
            try:
                os.chdir(self.dir)
                return os.path.abspath(self.job_parameters['working_directory'])
            finally:
                os.chdir(init_dir)
        else:
            return self.dir

    def __enter__(self):
        self._init_dir = os.getcwd()
        os.chdir(self.dir)
        self.checkpoint.__enter__()
        self.start = time.time()
        self.checkpoint['start'] = {
            'datetime': datetime.datetime.now().isoformat(),
            'timestamp': self.start
        }
        if self.parallelizer is None:
            self.parallelizer = Parallelizer.get_default()
        self.parallelizer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._init_dir = os.getcwd()
        os.chdir(self.dir)
        end = time.time()
        self.checkpoint['runtime'] = end - self.start
        self.checkpoint.__exit__(exc_type, exc_val, exc_tb)
        self.parallelizer.__exit__(exc_type, exc_val, exc_tb)


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
        if os.path.isdir(name):
            kw['initialization_directory'] = name
            name = os.path.basename(name)
        if timestamp:
            name += "_" + datetime.datetime.now().isoformat()
        return self.load(name, make_new=True, init=kw)