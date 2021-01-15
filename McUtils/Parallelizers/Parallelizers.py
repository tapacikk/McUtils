"""
Provides a simple framework for unifying different parallelism approaches.
Currently primarily targets multiprocessing and mpi4py, but also should work
with Ray. Dask will require more work unfortunately...
"""

import abc, weakref, functools, multiprocessing as mp, typing
import numpy as np

__all__ = [
    "Parallelizer",
    "MultiprocessingParallelizer",
    "MPIParallelizer",
    "SerialNonParallelizer"
]

class Parallelizer(metaclass=abc.ABCMeta):
    """
    Abstract base class to help manage parallelism.
    Provides the basic API that all parallelizers can be expected
    to conform to.
    Provides effectively the union of operations supported by
    `mp.Pool` and `MPI`.
    There is also the ability to lookup and register 'named'
    parallelizers, since we expect a single program to not
    really use more than one.
    This falls back gracefully to the serial case.
    """

    ####################################################################
    #  CONTEXT MANAGEMENT:
    #   details for initialization/registration/getting and setting
    #   up a parallelizer
    #
    parallelizer_registry = weakref.WeakValueDictionary()
    default_key="default"
    _default_par_stack = [] #stack to use when setting the default parallelizer

    def __init__(self):
        self._active_sentinel=0

    @classmethod
    def get_fallback_parallelizer(cls):
        return SerialNonParallelizer()

    @classmethod
    def lookup(cls, key):
        """
        Checks in the registry to see if a given parallelizer is there
        otherwise returns a `SerialNonParallelizer`.
        :param key:
        :type key:
        :return:
        :rtype:
        """
        if key is None:
            return cls.get_fallback_parallelizer()
        if key not in cls.parallelizer_registry:
            if key != cls.default_key:
                return cls.get_default()
            else:
                return cls.get_fallback_parallelizer()
        else:
            return cls.parallelizer_registry[key]
    @classmethod
    def get_default(cls):
        """
        Returns the 'default' parallelizer
        :return:
        :rtype:
        """
        return cls.lookup(cls.default_key)
    def register(self, key):
        """
        Checks in the registry to see if a given parallelizer is there
        otherwise returns a `SerialNonParallelizer`.
        :param key:
        :type key:
        :return:
        :rtype:
        """
        self.parallelizer_registry[key] = self
    def set_default(self):
        """
        Sets the parallelizer as the default one
        :return:
        :rtype:
        """
        if self.default_key in self.parallelizer_registry:
            self._default_par_stack.append(self.parallelizer_registry[self.default_key])
        self.register(self.default_key)
    def reset_default(self):
        """
        Resets the default parallelizer
        :return:
        :rtype:
        """
        if len(self._default_par_stack) > 0:
            self._default_par_stack.pop()

    @property
    def active(self):
        return self._active_sentinel > 0

    @abc.abstractmethod
    def initialize(self):
        """
        Initializes a parallelizer
        if necessary
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def finalize(self, exc_type, exc_val, exc_tb):
        """
        Finalizes a parallelizer (if necessary)
        if necessary
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    def __enter__(self):
        """
        Allows the parallelizer context to be set
        using `with`
        :return:
        :rtype:
        """
        if not self.active:
            self.set_default()
            self.initialize()
        self._active_sentinel += 1
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Allows the parallelizer context to be unset
        using `with`
        :param exc_type:
        :type exc_type:
        :param exc_val:
        :type exc_val:
        :param exc_tb:
        :type exc_tb:
        :return:
        :rtype:
        """
        self._active_sentinel -= 1
        if not self.active:
            self.reset_default()
            self.finalize(exc_type, exc_val, exc_tb)

    ####################################################################
    #  CONTROL FLOW:
    #   clean ways to manage flow of program
    #
    class InMainProcess:
        """
        Singleton representing being on the
        main process
        """
    class InWorkerProcess:
        """
        Singleton representing being on a
        worker process
        """
    @property
    @abc.abstractmethod
    def on_main(self):
        """
        Returns whether or not the executing process is the main
        process or not
        :return:
        :rtype:
        """

    @staticmethod
    def main_restricted(func):
        """
        A decorator to indicate that a function should only be
        run when on the main process
        :param func:
        :type func:
        :return:
        :rtype:
        """

        def main_process_func(*args, parallelizer=None, **kwargs):
            if parallelizer is None:
                parallelizer = Parallelizer.get_default()
            if parallelizer.on_main:
                kwargs['parallelizer'] = parallelizer
                return func(*args, **kwargs)
            else:
                return Parallelizer.InWorkerProcess

        return main_process_func

    @staticmethod
    def worker_restricted(func):
        """
        A decorator to indicate that a function should only be
        run when on a worker process
        :param func:
        :type func:
        :return:
        :rtype:
        """

        @functools.wraps(func)
        def worker_process_func(*args, parallelizer=None, **kwargs):
            if parallelizer is None:
                parallelizer = Parallelizer.get_default()
            if not parallelizer.on_main:
                kwargs['parallelizer'] = parallelizer
                return func(*args, **kwargs)
            else:
                return Parallelizer.InMainProcess

        return worker_process_func

    ####################################################################
    #  MPI API:
    #   Require core methods to be implemented to support the
    #   high-level MPI api.
    #   Currently only blocking operations are expected to be supported,
    #   since it's unlikely we'll ever have a use case for async calls
    #
    @abc.abstractmethod
    def send(self, data, loc, **kwargs):
        """
        Sends data to the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def receive(self, data, loc, **kwargs):
        """
        Receives data from the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def broadcast(self, data, **kwargs):
        """
        Sends the same data to all processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def scatter(self, data, **kwargs):
        """
        Performs a scatter of data to the different
        available parallelizer processes.
        *NOTE:* unlike in the MPI case, `data` does not
        need to be evenly divisible by the number of available
        processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @abc.abstractmethod
    def gather(self, data, **kwargs):
        """
        Performs a gather of data from the different
        available parallelizer processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    ####################################################################
    #  POOL API:
    #   require core methods to be implemented to act like an `mp.Pool`
    #
    @abc.abstractmethod
    def map(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    @abc.abstractmethod
    def starmap(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map with unpacking of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        raise NotImplementedError("Parallelizer is an abstract base class")

    @abc.abstractmethod
    def apply(self, func, *args, **kwargs):
        """
        Runs the callable `func` in parallel
        :param func:
        :type func:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    def run(self, func, *args, **kwargs):
        """
        Calls `apply`, but makes sure state is handled cleanly

        :param func:
        :type func:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        with self:
            return self.apply(func, *args, **kwargs)

    mode_map = {}
    @classmethod
    def from_config(cls,
                    mode=None,
                    **kwargs
                    ):
        if mode not in cls.mode_map:
            raise KeyError("don't know what to do with parallelization mode '{}'".format(
                mode
            ))
        par_cls = cls.mode_map[mode]
        return par_cls.from_config(**kwargs)

    ####################################################################
    #  UTILITY API:
    #   necessary things to make API more useable
    #
    @property
    def nprocs(self):
        """
        Returns the number of processes the parallelizer has
        to work with
        :return:
        :rtype:
        """
        return self.get_nprocs()
    @abc.abstractmethod
    def get_nprocs(self):
        """
        Returns the number of processes
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")
    @property
    def id(self):
        """
        Returns some form of identifier for the current process
        :return:
        :rtype:
        """
        return self.get_id()
    @abc.abstractmethod
    def get_id(self):
        """
        Returns the id for the current process
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

    def main_print(self, *args, **kwargs):
        """
        Prints from the main process
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        print(*args, **kwargs)
    def worker_print(self, *args, **kwargs):
        """
        Prints from a main worker process
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        print("On Worker {}:".format(self.id), *args, **kwargs)
    def print(self, *args, **kwargs):
        """
        An implementation of print that operates differently on workers than on main
        processes

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.on_main:
            return self.main_print(*args, **kwargs)
        else:
            return self.worker_print(*args, **kwargs)

    @abc.abstractmethod
    def wait(self):
        """
        Causes all processes to wait until they've met up at this point.
        :return:
        :rtype:
        """
        raise NotImplementedError("Parallelizer is an abstract base class")

class SendRecieveParallelizer(Parallelizer):
    """
    Parallelizer that implements `scatter`, `gather`, `broadcast`, and `map`
    based on just having a communicator that supports `send` and `receive methods
    """

    class SendReceieveCommunicator(metaclass=abc.ABCMeta):
        """
        A base class that provides an interface for
        sending/receiving data from a specific subprocesses
        """
        @property
        @abc.abstractmethod
        def locations(self):
            """
            Returns the list of locations known by the
            `SendReceieverCommunicator`
            :return:
            :rtype:
            """
        @property
        @abc.abstractmethod
        def location(self):
            """
            Returns the _current_ location
            :return:
            :rtype:
            """
        @abc.abstractmethod
        def send(self, data, loc, **kwargs):
            """
            Sends the specified data to loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
        @abc.abstractmethod
        def receive(self, data, loc, **kwargs):
            """
            Receives the specified data from loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
    @property
    @abc.abstractmethod
    def comm(self):
        """
        Returns the communicator used by the paralellizer
        :return:
        :rtype: SendRecieveParallelizer.SendReceieveCommunicator
        """
    def send(self, data, loc, **kwargs):
        """
        Sends data to the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.comm.send(data, loc, **kwargs)
    def receive(self, data, loc, **kwargs):
        """
        Receives data from the process specified by loc

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.comm.receive(data, loc, **kwargs)
    def broadcast(self, data, **kwargs):
        """
        Sends the same data to all processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.on_main:
            for i in self.comm.locations[1:]:
                self.comm.send(data, i, **kwargs)
            return data
        else:
            return self.comm.send(data, self.comm.location) # effectively a receive...
    def scatter(self, data, **kwargs):
        """
        Performs a scatter of data to the different
        available parallelizer processes.
        *NOTE:* unlike in the MPI case, `data` does not
        need to be evenly divisible by the number of available
        processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if self.on_main:
            locs = list(self.comm.locations) # gotta be safe
            nlocs = len(locs)
            chunk_size = len(data) // nlocs
            chunk_sizes = [chunk_size] * nlocs
            chunk_coverage = (chunk_size*nlocs)
            for i in range(len(data)-chunk_coverage):
                chunk_sizes[i] += 1
            s = chunk_sizes[0]
            main_data = data[:s]
            for i, b in zip(locs[1:], chunk_sizes[1:]):
                chunk = data[s:s+b]
                self.comm.send(chunk, i, **kwargs)
                s+=b
            return main_data
        else:
            return self.comm.send(data, self.comm.location) # effectively a receive...
    def gather(self, data, **kwargs):
        """
        Performs a gather of data from the different
        available parallelizer processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if self.on_main:
            locs = list(self.comm.locations) # gotta be safe
            nlocs = len(locs) # we reserve space for the main thread
            recv = [None] * nlocs
            recv[0] = data
            for n, i in enumerate(locs[1:]):
                res = self.comm.receive(data, i, **kwargs)
                if isinstance(res, Exception):
                    raise res
                recv[n+1] = res
            return recv
        else:
            return self.comm.receive(data, self.comm.location, **kwargs) # effectively a send...
    def map(self, func, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        # self.wait()
        # self.print("Scattering Data")
        data = self.scatter(data, **kwargs)
        # self.wait()
        # self.print("Broadcasting Extra Args")
        extra_args = self.broadcast(extra_args, **kwargs)
        # self.wait()
        # self.print("Broadcasting Extra Kwargs")
        extra_kwargs = self.broadcast(extra_kwargs, **kwargs)
        if extra_args is None:
            extra_args = ()
        if extra_kwargs is None:
            extra_kwargs = {}
        # try:
        evals = [func(sub_data, *extra_args, **extra_kwargs) for sub_data in data]
        # except Exception as e:
        #     self.gather(e, **kwargs)
        #     raise
        res = self.gather(evals, **kwargs)
        if self.on_main:
            return sum(res, [])
        else:
            return Parallelizer.InWorkerProcess
    def starmap(self, func, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a parallel map with unpacking of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        data = self.scatter(data, **kwargs)
        extra_args = self.broadcast(extra_args)
        extra_kwargs = self.broadcast(extra_kwargs)
        if extra_args is None:
            extra_args = ()
        if extra_kwargs is None:
            extra_kwargs = {}
        # try:
        evals = [func(*sub_data, *extra_args, **extra_kwargs) for sub_data in data]
        # except Exception as e:
        #     self.gather(e, **kwargs)
        #     raise
        res = self.gather(evals, **kwargs)
        if self.on_main:
            return sum(res, [])
        else:
            return Parallelizer.InWorkerProcess

    def wait(self):
        """
        Causes all processes to wait until they've met up at this point.
        :return:
        :rtype:
        """

        # flag = self.broadcast("SyncFlag")
        self.gather("SyncFlag") # sends data to main
        self.broadcast("SyncFlag")

class MultiprocessingParallelizer(SendRecieveParallelizer):
    """
    Parallelizes using a  process pool and a runner
    function that represents a "main loop".
    """

    class SendRecvQueuePair:
        def __init__(self, id:int, manager:'mp.managers.SyncManager'):
            self.id=id
            self.send_queue = manager.Queue()
            self.receive_queue = manager.Queue()
            self.init_flag = manager.Event()
    class PoolCommunicator(SendRecieveParallelizer.SendReceieveCommunicator):
        """
        Defines a serializable process communicator
        that allows communication with a managed `mp.Pool`
        to support `send` and `receive` and therefore to
        support the rest of the necessary bits of the MPI API
        """
        initialization_timeout=.5
        def __init__(self,
                     parent: 'MultiprocessingParallelizer',
                     id:int,
                     queues: typing.Iterable['MultiprocessingParallelizer.SendRecvQueuePair']
                     ):
            self.parent = parent
            self.id = id
            self.queues = tuple(queues)

        class PoolError(Exception):
            """
            For errors arising from Pool operations
            """
            pass

        def initialize(self):
            """
            Performs initialization of the communicator
            (basically just waits until all threads say all is well)
            :return:
            :rtype:
            """
            self.queues[self.id].init_flag.set()
            if self.parent.on_main:
                for q in self.queues:
                    wat = q.init_flag.wait(self.initialization_timeout)
                    if not wat:
                        raise self.PoolError("Failed to initialize pool")

        @property
        def locations(self):
            """
            Returns the list of queue ids
            :return:
            :rtype:
            """
            return [x.id for x in self.queues]
        @property
        def location(self):
            """
            Returns the _current_ location
            :return:
            :rtype:
            """
            return self.id

        def send(self, data, loc, **kwargs):
            """
            Sends the specified data to loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            queue = self.queues[loc].send_queue #type: mp.queues.Queue
            if loc == self.id:
                # print("Send", "<<<<", self.id)
                res = queue.get()
                # print("Sent to", self.id)
                return res
            else:
                # print("Send", self.id, ">>>>", loc)
                queue.put(data)
                return data
        def receive(self, data, loc, **kwargs):
            """
            Receives the specified data from loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            queue = self.queues[loc].receive_queue
            if loc != self.id:
                # print("Recv", loc, ">>>>", self.id)
                res = queue.get()
                # print("Recieved on", self.id, "from", loc)
                return res
            else:
                # print("Recv", "<<<<", self.id)
                queue.put(data)
                return data

    def __init__(self,
                 worker=False,
                 pool:mp.Pool=None,
                 context=None,
                 manager=None,
                 **kwargs
                 ):
        super().__init__()
        self.opts=kwargs
        self.pool=pool
        self.worker=worker
        self.ctx=context
        self.manager=manager
        self._comm = None
        self._comm_list = None
        self.nproc = None

    def get_nprocs(self):
        return self.nproc
    def get_id(self):
        return self.comm.id

    @property
    def comm(self):
        """
        Returns the communicator used by the paralellizer
        :return:
        :rtype: MultiprocessingParallelizer.PoolCommunicator
        """
        return self._comm

    def __getstate__(self):
        # most things don't need to be mapped over...
        state = self.__dict__.copy()
        state['pool'] = None
        state['worker'] = True
        state['manager'] = None
        state['comm'] = None
        state['_comm'] = None
        state['_comm_list'] = None
        state['queues'] = None
        # print(state)
        # state = {
        #     'opts': self.opts,
        #     'pool': None,
        #     'worker': True,
        #     'manager': None, #self.manager,
        #     'comm': None,
        #     'nproc': self.nproc
        # }
        # print(self.runner)
        # state['runner'] = None
        # state['comm'] = None
        # state['worker'] = True # so that the worker processes know they're workers
        # print(state)
        return state

    @staticmethod
    def _run(runner, comm:PoolCommunicator, args, kwargs):
        """
        Static runner function that just dispatches methods out to
        cores
        :param runner:
        :type runner:
        :param comm:
        :type comm:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        self=comm.parent
        with self:
            self._comm = comm # makes a cyclic dependency...but oh well
            # self.print("...?")
            self._comm.initialize()
            # self.print("fack")
            kwargs['parallelizer'] = comm.parent
            # self.print(runner)
            return runner(*args, **kwargs)

    def apply(self, func, *args, **kwargs):
        """
        Applies func to args in parallel on all of the processes

        :param func:
        :type func:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        import multiprocessing as mp

        if self._comm_list is None:
            self._comm_list = [self.PoolCommunicator(self, i, self.queues) for i in range(0, self.nproc+1)]
        self._comm = self._comm_list[0]
        mapping = list(zip(
                [func] * self.nproc,
                self._comm_list[1:],
                [args] * self.nproc,
                [kwargs] * self.nproc
            ))
        pool = self.pool #type: mp.pool.Pool
        subsidiary = pool.starmap_async(self._run, mapping)
        try:
            main = self._run(func, self.PoolCommunicator(self, 0, self.queues), args, kwargs)
        except self.PoolCommunicator.PoolError:
            # check for errors on subsidiary...
            subsidiary.get(timeout=.5)
            raise
        subs = subsidiary.get() # just to effect a wait
        return main

    def _get_pool(self,
                  manager: mp.Manager,
                  **kwargs
                  ):
        if 'processes' not in kwargs:
            kwargs['processes'] = mp.cpu_count() - 1
        return manager.Pool(**kwargs)

    @staticmethod
    def get_pool_context(pool):
        return pool._ctx # I don't like doing this but seems like only way?
    @staticmethod
    def get_pool_nprocs(pool):
        return pool._processes  # see above

    def initialize(self):
        if not self.worker:
            if self.pool is None:
                if self.ctx is None:
                    self.ctx = mp.get_context() # get the default context
                self.pool = self._get_pool(self.ctx, **self.opts)
            elif self.ctx is None:
                self.ctx = self.get_pool_context(self.pool)
            if self.manager is None:
                self.manager = mp.Manager()
            self.pool.__enter__()
            self.nproc = self.get_pool_nprocs(self.pool)
            self.queues = [self.SendRecvQueuePair(i, self.manager) for i in range(0, self.nproc+1)]

    def finalize(self, exc_type, exc_val, exc_tb):
        if not self.worker:
            self.pool.__exit__(exc_type, exc_val, exc_tb)
            self.queues = None
            self._comm = None
    @property
    def on_main(self):
        return not self.worker

    @classmethod
    def from_config(cls, **kw):
        return cls(**kw)
Parallelizer.mode_map['multiprocessing'] = MultiprocessingParallelizer

class MPIParallelizer(SendRecieveParallelizer):
    """
    Parallelizes using `mpi4py`
    """

    class MPICommunicator(SendRecieveParallelizer.SendReceieveCommunicator):
        """
        A base class that provides an interface for
        sending/receiving data from a specific subprocesses
        """

        def __init__(self, parent, mpi_comm):
            self.parent = parent
            self.comm = mpi_comm
        @property
        def nprocs(self):
            """
            Returns the number of available processes
            :return:
            :rtype:
            """
            return self.comm.Get_size()
        @property
        def locations(self):
            """
            Returns the list of locations known by the
            `SendReceieverCommunicator`
            :return:
            :rtype:
            """
            return range(self.comm.Get_size())
        @property
        def location(self):
            """
            Returns the _current_ location
            :return:
            :rtype:
            """
            return self.comm.Get_rank()
        def send(self, data, loc, **kwargs):
            """
            Sends the specified data to loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if loc == self.location:
                return self.comm.receive(source=self.comm.ANY_LOCATION)
            else:
                self.comm.send(data, dest=loc)
        def receive(self, data, loc, root=0, **kwargs):
            """
            Receives the specified data from loc
            :param data:
            :type data:
            :param loc:
            :type loc:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if loc != self.location:
                self.comm.send(self.location, dest=loc)
                return self.comm.receive(source=loc)
            else:
                where_to = self.comm.recv(source=self.comm.ANY_LOCATION)
                self.comm.send(data, dest=where_to)
        def broadcast(self, data, root=0, **kwargs):
            """
            Sends the same data to all processes

            :param data:
            :type data:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if isinstance(data, np.ndarray):
                return self.comm.Bcast(data, root=root)
            else:
                return self.comm.bcast(data, root=root)
        def scatter_obj(self, data, root=0, **kwargs):
            """
            Scatters data to the different MPI ranks using a series
            of send calls.
            This is the default for anything except numpy arrays.

            :param data:
            :type data:
            :param root:
            :type root:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if self.location == root:
                locs = list(self.locations)  # gotta be safe
                nlocs = len(locs)
                chunk_size = len(data) // nlocs
                chunk_sizes = [chunk_size] * nlocs
                for i in range(len(data) - (chunk_size*nlocs)):
                    chunk_sizes[i] += 1

                s = chunk_sizes[0]
                main_data = data[:s]
                for i, b in zip(locs[1:], chunk_sizes[1:]):
                    chunk = data[s:s + b]
                    self.send(chunk, i, **kwargs)
                    s += b
                return main_data
        def scatter(self, data, root=0, shape=None, **kwargs):
            """
            Performs a scatter of data to the different
            available parallelizer processes.
            *NOTE:* unlike in the MPI case, `data` does not
            need to be evenly divisible by the number of available
            processes

            :param data:
            :type data:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """

            if isinstance(data, np.ndarray) or data is None:
                if root == self.location and data is None:
                    raise TypeError("'None' is not scatterable. Try `broadcast`.")
                if root == self.location:
                    send_buf = np.ascontiguousarray(data)
                else:
                    send_buf = None
                if shape is None:
                    if root == self.location:
                        shape = data.shape
                    shape = self.broadcast(shape)
                ranks = self.comm.Get_size()
                ndat = shape[0]
                block_size = ndat // ranks
                block_remainder = ndat - (block_size*ranks)
                if block_remainder == 0:
                    shape = (block_size,) + shape[1:]
                    recv_buf = np.empty(shape, dtype=data.dtype)
                    self.comm.Scatter(send_buf, recv_buf, root=root)
                    return recv_buf
                else:
                    # it turns out MPI4Py has only an unsophisticated
                    # implementation of Scatterv that explicitly needs the
                    # appropriate block offsets, since it basically expects
                    # to have a flattened form of the array (just like MPI)
                    block_sizes = [block_size]*ranks
                    for i in range(block_remainder):
                        block_sizes[i] += 1
                    block_sizes = np.array(block_sizes)
                    block_offset = int(np.prod(data.shape[1:]))
                    block_offsets = np.cumsum(block_sizes*block_offset)
                    recv_buf = np.empty((block_sizes[self.location],) + data.shape[1:], dtype=data.dtype)
                    return self.comm.Scatterv(
                        send_buf,
                        block_sizes,
                        block_offsets,
                        recv_buf,
                        root=root
                     )
            else:
                return self.scatter_obj(data, root=root, **kwargs)
        def gather_obj(self, data, root=0, **kwargs):
            if self.location == root:
                locs = list(self.locations)  # gotta be safe
                nlocs = len(locs)  # we reserve space for the main thread
                recv = [None] * nlocs
                recv[0] = data
                for n, i in enumerate(locs[1:]):
                    recv[n + 1] = self.comm.receive(data, i, **kwargs)
                return recv
            else:
                return self.receive(data, self.location, **kwargs)  # effectively a send...
        def gather(self, data, root=0, shape=None, **kwargs):
            """
            Performs a gather from the different
            available parallelizer processes.
            *NOTE:* unlike in the MPI case, `data` does not
            need to be evenly divisible by the number of available
            processes

            :param data:
            :type data:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:
            """
            if isinstance(data, np.ndarray):
                # make sure everyone knows what's up
                send_buf = np.ascontiguousarray(data)
                if shape is None:
                    block_sizes = self.gather(data.shape[0], root=root)
                    ndat = int(np.sum(block_sizes))
                    if root == self.location:
                        shape = (ndat,) + data.shape
                    shape = self.broadcast(shape)
                # otherwise send shit around
                ranks = self.comm.Get_size()
                ndat = shape[0]
                block_size = ndat // ranks
                block_remainder = ndat - (block_size*ranks)
                if block_remainder == 0:
                    if root == self.location:
                        recv_buf = np.empty(shape, dtype=data.dtype)
                    else:
                        recv_buf = None
                    self.comm.Gather(send_buf, recv_buf, root=root)
                    return recv_buf
                else:
                    block_sizes = [block_size] * ranks
                    for i in range(block_remainder):
                        block_sizes[i] += 1
                    block_sizes = np.array(block_sizes)
                    block_offset = int(np.prod(data.shape[1:]))
                    block_offsets = np.cumsum(block_sizes * block_offset)
                    if root == self.location:
                        recv_buf = np.empty(shape, dtype=data.dtype)
                    else:
                        recv_buf = None
                    return self.comm.Gatherv(
                        send_buf,
                        block_sizes,
                        block_offsets,
                        recv_buf,
                        root=root
                    )
            else:
                return self.gather_obj(data, root=root, **kwargs)

    def __init__(self, root=0, comm=None):
        super().__init__()

        from mpi4py import MPI as api
        self.api = api
        if comm is None:
            comm = api.COMM_WORLD
        self.root = root
        self._comm = self.MPICommunicator(self, comm)
        self.world_size = None
        self.world_rank = None

    def get_nprocs(self):
        return self._comm.nprocs
    def get_id(self):
        return self._comm.location

    def initialize(self):
        # handled by mpi4py?
        pass
    def finalize(self, exc_type, exc_val, exc_tb):
        # handled by mpi4py?
        pass

    @property
    def comm(self):
        """
        Returns the communicator used by the paralellizer
        :return:
        :rtype: MPIParallelizer.MPICommunicator
        """
        return self._comm

    @property
    def on_main(self):
        return self.comm.location == 0

    def broadcast(self, data, **kwargs):
        """
        Sends the same data to all processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.comm.broadcast(data, root=self.root, **kwargs)
    def scatter(self, data, shape=None, **kwargs):
        """
        Performs a scatter of data to the different
        available parallelizer processes.
        *NOTE:* unlike in the MPI case, `data` does not
        need to be evenly divisible by the number of available
        processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.comm.scatter(data, root=self.root, shape=shape, **kwargs)
    def gather(self, data, shape=None, **kwargs):
        """
        Performs a gather of data from the different
        available parallelizer processes

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return self.comm.gather(data, root=self.root, shape=shape, **kwargs)
    def map(self, func, data, input_shape=None, output_shape=None, **kwargs):
        """
        Performs a parallel map of function over
        the held data on different processes

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        sub_data = self.scatter(data, shape=input_shape, **kwargs)
        res = func(sub_data)
        return self.gather(res, shape=output_shape, **kwargs)

    def apply(self, func, *args, **kwargs):
        """
        Applies func to args in parallel on all of the processes.
        For MPI, since jobs are always started with mpirun, this
        is just a regular apply

        :param func:
        :type func:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return func(*args, parallelizer=self, **kwargs)

    @classmethod
    def from_config(cls, **kw):
        return cls(**kw)
Parallelizer.mode_map['mpi'] = MPIParallelizer

class SerialNonParallelizer(Parallelizer):
    """
    Totally serial evaluation for cases where no parallelism
    is provide
    """

    def get_nprocs(self):
        return 1
    def get_id(self):
        return 0

    def initialize(self):
        """
        Initializes a parallelizer
        if necessary
        :return:
        :rtype:
        """
        pass
    def finalize(self, exc_type, exc_val, exc_tb):
        """
        Finalizes a parallelizer (if necessary)
        if necessary
        :return:
        :rtype:
        """
        pass
    @property
    def on_main(self):
        """
        Returns whether or not the executing process is the main
        process or not
        :return:
        :rtype:
        """
        return True
    def send(self, data, loc, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def receive(self, data, loc, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def broadcast(self, data, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def scatter(self, data, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data
    def gather(self, data, **kwargs):
        """
        A no-op

        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return data

    def map(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a serial map of the function over
        the passed data

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if extra_args is not None or extra_kwargs is not None:
            if extra_args is None:
                extra_args=()
            if extra_kwargs is None:
                extra_kwargs={}
            function = lambda a, fn=function, ea=extra_args, ek=extra_kwargs: fn(a, *ea, **ek)

        return list(map(function, data, **kwargs))

    def starmap(self, function, data, extra_args=None, extra_kwargs=None, **kwargs):
        """
        Performs a serial map with unpacking of the function over
        the passed data

        :param function:
        :type function:
        :param data:
        :type data:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        if extra_args is None:
            extra_args=()
        if extra_kwargs is None:
            extra_kwargs={}
        function = lambda a, fn=function, ea=extra_args, ek=extra_kwargs: fn(*a, *ea, **ek)

        return list(map(function, data, **kwargs))

    def apply(self, func, *args, **kwargs):
        kwargs['parallelizer'] = self
        return func(*args, **kwargs)

    def wait(self):
        """
        No need to wait when you're in a serial environment
        :return:
        :rtype:
        """

        pass



