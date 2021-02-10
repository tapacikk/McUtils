
import abc, os
from .Serializers import *

__all__ = [
    "Checkpointer",
    "DumpCheckpointer",
    "JSONCheckpointer",
    "NumPyCheckpointer",
    "HDF5Checkpointer",
    "NullCheckpointer"
]

class Checkpointer(metaclass=abc.ABCMeta):
    """
    General purpose base class that allows checkpointing to be done easily and cleanly.
    Intended to be a passable object that allows code to checkpoint easily.
    """
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self._came_open = not isinstance(checkpoint_file, str)
        self._stream = None

    _ext_map = None
    @classmethod
    def extension_map(cls):
        if cls._ext_map is not None:
            return cls._ext_map
        else:
            return {
                '.json':JSONCheckpointer,
                '.hdf5':HDF5Checkpointer,
                '.npz':NumPyCheckpointer
                # ('.yaml','.yml'):YA
            }

    @classmethod
    def from_file(cls, file, **opts):
        """
        Dispatch function to load from the appropriate file
        :param file:
        :type file: str | File
        :param opts:
        :type opts:
        :return:
        :rtype:
        """

        if not isinstance(file, str):
            #TODO: make this cleaner
            file_name = file.name # might break in the future...
        else:
            file_name = file

        _, ext = os.path.splitext(file_name)

        ext_map = cls.extension_map()

        if ext not in ext_map:
            raise ValueError("don't know have default checkpointer type registered for extension {}".format(ext))

        return ext_map[ext](file, **opts)

    def __enter__(self):
        if self._stream is None:
            self._stream = self.open_checkpoint_file(self.checkpoint_file)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream is not None:
            self.close_checkpoint_file(self._stream)
            self._stream = None

    @property
    def is_open(self):
        return self._stream is not None

    @property
    def stream(self):
        return self._stream

    @abc.abstractmethod
    def open_checkpoint_file(self, chk):
        """
        Opens the passed `checkpoint_file` (if not already open)
        :param chk:
        :type chk:
        :return:
        :rtype:
        """
        raise NotImplementedError("CheckpointerBase is an abstract base class...")
    @abc.abstractmethod
    def close_checkpoint_file(self, stream):
        """
        Closes the opened checkpointing stream
        :param stream:
        :type stream:
        :return:
        :rtype:
        """
        raise NotImplementedError("CheckpointerBase is an abstract base class...")
    @abc.abstractmethod
    def save_parameter(self, key, value):
        """
        Saves a parameter to the checkpoint file
        :param key:
        :type key:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        raise NotImplementedError("CheckpointerBase is an abstract base class...")
    @abc.abstractmethod
    def load_parameter(self, key):
        """
        Loads a parameter from the checkpoint file
        :param key:
        :type key:
        :return:
        :rtype:
        """
        raise NotImplementedError("CheckpointerBase is an abstract base class...")

    def __getitem__(self, item):
        return self.load_parameter(item)
    def __setitem__(self, key, value):
        self.save_parameter(key, value)

    @abc.abstractmethod
    def keys(self):
        """
        Returns the keys of currently checkpointed
        objects

        :return:
        :rtype:
        """
        raise NotImplementedError("Checkpointer is an abstract base class...")


class DumpCheckpointer(Checkpointer):
    """
    A subclass of `CheckpointerBase` that writes an entire dump to file at once & maintains
    a backend cache to update it cleanly
    """
    def __init__(self, file, cache=None, open_kwargs=None):
        self.backend = cache # cache values
        super().__init__(file)
        if open_kwargs is None:
            open_kwargs = {'mode':"w+"}
        self.open_kwargs = open_kwargs
    def load_cache(self):
        if self.backend is None:
            self.backend = {}
    def __enter__(self):
        self.load_cache()
        return super().__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.dump()
        finally:
            super().__exit__(exc_type, exc_val, exc_tb)
    @abc.abstractmethod
    def dump(self):
        """
        Writes the entire data structure
        :return:
        :rtype:
        """
        raise NotImplementedError("DumpCheckpointer is an ABC and doesn't know how to write to file")
    def convert(self):
        """
        Converts the cache to an exportable form if needed
        :return:
        :rtype:
        """
        return self.backend
    def open_checkpoint_file(self, chk):
        """
        Opens the passed `checkpoint_file` (if not already open)
        :param chk:
        :type chk:
        :return:
        :rtype:
        """
        if isinstance(chk, str):
            chk = open(chk, **self.open_kwargs)
        return chk
    def close_checkpoint_file(self, stream):
        """
        Closes the opened checkpointing stream
        :param stream:
        :type stream:
        :return:
        :rtype:
        """
        if not self._came_open:
            stream.close()
    def save_parameter(self, key, value):
        """
        Saves a parameter to the checkpoint file
        :param key:
        :type key:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        self.backend[key] = value
    def load_parameter(self, key):
        """
        Loads a parameter from the checkpoint file
        :param key:
        :type key:
        :return:
        :rtype:
        """
        return self.backend[key]

    def keys(self):
        return self.backend.keys()

class JSONCheckpointer(DumpCheckpointer):
    """
    A checkpointer that uses JSON as a backend
    """

    def __init__(self, file, cache=None, serializer=None, open_kwargs=None):
        if serializer is None:
            serializer = JSONSerializer()
        self.serializer = serializer
        super().__init__(file, cache=cache, open_kwargs=open_kwargs)

    def load_cache(self):
        cache = self.backend
        if cache is None:
            file = self.checkpoint_file
            serializer = self.serializer
            if isinstance(file, str) and os.path.exists(file) and os.stat(file).st_size > 0:
                with open(file, 'r') as stream:
                    cache = serializer.deserialize(stream)
                if not isinstance(cache, dict):
                    cache = {}
            else:
                cache = {}
            self.backend = cache

    def dump(self):
        """
        Writes the entire data structure
        :return:
        :rtype:
        """
        self.serializer.serialize(self.stream, self.backend)

class NumPyCheckpointer(DumpCheckpointer):
    """
    A checkpointer that uses NumPy as a backend
    """

    def __init__(self, file, cache=None, serializer=None, open_kwargs=None):
        if isinstance(file, str):
            if not os.path.exists(file):
                if os.path.exists(file + '.npz'):
                    file = file + '.npz'
                elif os.path.exists(file + '.npy'):
                    file = file + '.npy'

        if serializer is None:
            serializer = NumPySerializer()
        self.serializer = serializer
        if open_kwargs is None:
            open_kwargs = {'mode':'bw'}
        super().__init__(file, cache=cache, open_kwargs=open_kwargs)

    def load_cache(self):
        cache = self.backend
        if cache is None:
            file = self.checkpoint_file
            serializer = self.serializer
            if isinstance(file, str) and os.path.exists(file) and os.stat(file).st_size > 0:
                with open(file, 'br') as stream:
                    cache = serializer.deserialize(stream)
                if not isinstance(cache, dict):
                    cache = {}
            else:
                cache = {}
            self.backend = cache

    def dump(self):
        """
        Writes the entire data structure
        :return:
        :rtype:
        """
        self.serializer.serialize(self.stream, self.backend)

class HDF5Checkpointer(Checkpointer):
    """
    A checkpointer that uses an HDF5 file as a backend.
    Doesn't maintain a secondary `dict`, because HDF5 is an updatable format.
    """

    def __init__(self, checkpoint_file, serializer=None):
        super().__init__(checkpoint_file)
        if serializer is None:
            serializer = HDF5Serializer()
        self.serializer = serializer

    def open_checkpoint_file(self, chk):
        """
        Opens the passed `checkpoint_file` (if not already open)
        :param chk:
        :type chk: str | file-like
        :return:
        :rtype:
        """
        if not self._came_open:
            if os.path.exists(chk):
                return open(chk, 'a+b')
            else:
                return open(chk, "w+b")
        elif 'b' not in chk.mode:
            raise IOError("{} isn't opened in binary mode (HDF5 needs that)".format(chk))

    def close_checkpoint_file(self, stream):
        """
        Opens the passed `checkpoint_file` (if not already open)
        :param chk:
        :type chk:
        :return:
        :rtype:
        """
        if not self._came_open:
            self.stream.close()

    def save_parameter(self, key, value):
        """
        Saves a parameter to the checkpoint file
        :param key:
        :type key:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        # HDF5 serialization is an updateable process
        self.serializer.serialize(self.stream, {key:value})

    def load_parameter(self, key):
        """
        Loads a parameter from the checkpoint file
        :param key:
        :type key:
        :return:
        :rtype:
        """
        return self.serializer.deserialize(self.stream, key=key)

    def keys(self):
        fff = self.serializer.api.File(self.stream)
        return fff.keys()

class NullCheckpointer(Checkpointer):
    """
    A checkpointer that doesn't actually do anything, but which is provided
    so that programs can turn off checkpointing without changing their layout
    """
    def __init__(self, checkpoint_file=None):
        super().__init__(checkpoint_file)
        self.backend = {}

    def open_checkpoint_file(self, chk):
        """
        Opens the passed `checkpoint_file` (if not already open)
        :param chk:
        :type chk: str | file-like
        :return:
        :rtype:
        """
        return "NotAFile"

    def close_checkpoint_file(self, stream):
        """
        Opens the passed `checkpoint_file` (if not already open)
        :param chk:
        :type chk:
        :return:
        :rtype:
        """
        pass

    def save_parameter(self, key, value):
        """
        Saves a parameter to the checkpoint file
        :param key:
        :type key:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        self.backend[key] = value

    def load_parameter(self, key):
        """
        Loads a parameter from the checkpoint file
        :param key:
        :type key:
        :return:
        :rtype:
        """
        return self.backend[key]
