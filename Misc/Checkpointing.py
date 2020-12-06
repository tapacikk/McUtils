
import abc

__all__ = [
    "CheckpointerBase"
]

class CheckpointerBase(metaclass=abc.ABCMeta):
    """
    General purpose base class that allows checkpointing to be done easily and cleanly.
    Intended to be a passable object that allows code to checkpoint easily.
    """
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self._stream = None

    def __enter__(self):
        if self._stream is None:
            self._stream = self.open_checkpoint_file(self.checkpoint_file)
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream is not None:
            self.close_checkpoint_file(self._stream)
            self._stream = None

    @property
    def is_open(self):
        return self._stream is not None

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

class DumpCheckpointer(CheckpointerBase):
    """
    A subclass of `CheckpointerBase` that writes an entire dump to file at once & maintains
    a backend cache to update it cleanly
    """
    def __init__(self, file, cache=None):
        if cache is None:
            cache = {}
        self.backend = cache # cache values
        super().__init__(file)
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
            chk = open(chk, "w+")
        return chk
    def close_checkpoint_file(self, stream):
        """
        Closes the opened checkpointing stream
        :param stream:
        :type stream:
        :return:
        :rtype:
        """
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

class JSONCheckpointer(DumpCheckpointer):
    """
    A checkpointer that uses JSON as a backend
    """

    json_types  = (str, int, float) # directly JSON-serializable types
    @classmethod
    def _convert(cls, data):
        """
        Recursively loop through, test data,
        :param data:
        :type data:
        :return:
        :rtype:
        """
        if isinstance(data, cls.json_types):
            return data
        if isinstance(data, dict):
            ...
        elif isinstance():
            ...


    def convert(self):
        """
        Walks through the backend and converts data structures to JSON-able forms
        :return:
        :rtype:
        """
        duplicate = {}
        for

