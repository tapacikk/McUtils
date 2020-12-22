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
    _default_par_stack = [] #stack to use when setting the default parallelizer
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
        if key not in cls.parallelizer_registry:
            return SerialNonParallelizer()
        else:
            return cls.parallelizer_registry[key]
    @classmethod
    def get_default(cls):
        """
        Returns the 'default' parallelizer
        :return:
        :rtype:
        """
        return cls.lookup('default')
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
        if 'default' in self.parallelizer_registry:
            self._default_par_stack.append(self.parallelizer_registry['default'])
        self.register('default')
    def reset_default(self):
        """
        Resets the default parallelizer
        :return:
        :rtype:
        """
        if len(self._default_par_stack) > 0:
            self._default_par_stack.pop()

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
        self.set_default()
        self.initialize()
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
                return func(*args, parallelizer=parallelizer, **kwargs)
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
                return func(*args, parallelizer=parallelizer, **kwargs)
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
    def map(self, function, data, **kwargs):
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
        with self:
            return self.apply(
                func,
                *args,
                **kwargs
            )

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
            for i in self.comm.locations:
                self.comm.send(data, i, **kwargs)
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
            chunk_remainder = len(data) % chunk_size
            if chunk_remainder == 0:
                chunk_sizes = [chunk_size] * nlocs
            else:
                chunk_sizes = [chunk_size] * (nlocs - 1) + [chunk_size + chunk_remainder]

            main_data = data[:chunk_sizes[0]]
            s = chunk_size
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
                recv[n+1] = self.comm.receive(data, i, **kwargs)
            return recv
        else:
            return self.comm.receive(data, self.comm.location, **kwargs) # effectively a send...
    def map(self, func, data, **kwargs):
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
        sub_data = self.scatter(data, **kwargs)
        res = func(sub_data)
        return self.gather(res, **kwargs)

class MultiprocessingParallelizer(SendRecieveParallelizer):
    """
    Parallelizes using a  process pool and a runner
    function that represents a "main loop".
    """

    class SendRecvQueuePair:
        def __init__(self, id:int, manager:mp.Manager):
            self.id=id
            self.send_queue = manager.Queue()
            self.receive_queue = manager.Queue()
    class PoolCommunicator(SendRecieveParallelizer.SendReceieveCommunicator):
        """
        Defines a serializable process communicator
        that allows communication with a managed `mp.Pool`
        to support `send` and `receive` and therefore to
        support the rest of the necessary bits of the MPI API
        """
        def __init__(self,
                     parent: 'MultiprocessingParallelizer',
                     id:int,
                     queues: typing.Iterable['MultiprocessingParallelizer.SendRecvQueuePair']
                     ):
            self.parent = parent
            self.id = id
            self.queues = tuple(queues)

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
            queue = self.queues[loc].send_queue
            # print(">>>>", self.id, loc)
            if loc == self.id:
                return queue.get()
            else:
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
            # print("<<<<", self.id, loc)
            if loc != self.id:
                return queue.get()
            else:
                queue.put(data)
                return data

    def __init__(self,
                 worker=False,
                 pool:mp.Pool=None,
                 context=None,
                 manager=None,
                 **kwargs
                 ):
        self.opts = kwargs
        self.pool = pool
        self.worker = worker
        self.ctx = context
        self.manager=manager
        self._comm = None
        self.nproc = None
    @property
    def comm(self):
        """
        Returns the communicator used by the paralellizer
        :return:
        :rtype: MultiprocessingParallelizer.PoolCommunicator
        """
        return self._comm

    def __getstate__(self):
        state = self.opts.copy()
        # the mapped process doesn't need to know about either of these
        state['runner'] = None
        state['comm'] = None
        state['worker'] = True # so that the worker processes know they're workers
        return state

    @staticmethod
    def _run(runner, comm:PoolCommunicator, args, kwargs):
        self=comm.parent
        with self:
            self._comm = comm # makes a cyclic dependency...but oh well
            kwargs['parallelizer'] = comm.parent
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
        self._comm = self.PoolCommunicator(self, 0, self.queues)
        # self.pool = mp.Pool()
        subsidiary = self.pool.starmap_async(
            self._run,
            zip(
                [func] * self.nproc,
                [self.PoolCommunicator(self, i, self.queues) for i in range(1, self.nproc + 1)],
                [args] * self.nproc,
                [kwargs] * self.nproc
            )
        )
        main = self._run(func, self.PoolCommunicator(self, 0, self.queues), args, kwargs)
        subs = subsidiary.get() # just to effect a wait
        return main

    def _get_pool(self,
                  manager: mp.Manager,
                  **kwargs
                  ):
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
                chunk_remainder = len(data) % chunk_size
                if chunk_remainder == 0:
                    chunk_sizes = [chunk_size] * nlocs
                else:
                    chunk_sizes = [chunk_size] * (nlocs - 1) + [chunk_size + chunk_remainder]

                main_data = data[:chunk_sizes[0]]
                s = chunk_size
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
                block_remainder = ndat % ranks
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
                    block_offset = int(np.prod(data.shape[1:]))
                    block_sizes = np.array([block_size]*ranks + [block_remainder])
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
                block_remainder = ndat % ranks
                if block_remainder == 0:
                    if root == self.location:
                        recv_buf = np.empty(shape, dtype=data.dtype)
                    else:
                        recv_buf = None
                    self.comm.Gather(send_buf, recv_buf, root=root)
                    return recv_buf
                else:
                    block_offset = int(np.prod(data.shape[1:]))
                    block_sizes = np.array([block_size] * ranks + [block_remainder])
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
        from mpi4py import MPI as api
        self.api = api
        if comm is None:
            comm = api.COMM_WORLD
        self.root = root
        self._comm = self.MPICommunicator(self, comm)
        self.world_size = None
        self.world_rank = None

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
        return func(*args, **kwargs)

    @classmethod
    def from_config(cls, **kw):
        return cls(**kw)
Parallelizer.mode_map['mpi'] = MPIParallelizer

class SerialNonParallelizer(Parallelizer):
    """
    Totally serial evaluation for cases where no parallelism
    is provide
    """

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
    def map(self, function, data, **kwargs):
        """
        Performs a serial map of function over
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
        return map(function, data, **kwargs)

    def apply(self, func, *args, **kwargs):
        kwargs['parallelizer'] = self
        return func(*args, **kwargs)



