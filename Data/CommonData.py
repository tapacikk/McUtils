"""
Defines a common data handler
"""
import os, sys

__all__ = [ "DataHandler", "DataError" ]

default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
default_data_package = "TheRealMcCoy"
default_data_key = "data"
default_data_source_key = "source"

class DataError(KeyError):
    """
    Exception subclass for data error
    """

class DataHandler:
    """
    Defines a general data loader class that we can use for AtomData and any other data classes we might find useful
    """
    def __init__(self,
                 data_name,
                 data_key = None,
                 source_key = None,
                 data_dir = None,
                 data_pkg = None,
                 alternate_keys = None,
                 getter = None
                 ):
        if data_dir is None:
            data_dir = default_data_dir
        if data_key is None:
            data_key = default_data_key
        if source_key is None:
            source_key = default_data_source_key
        if data_pkg is None:
            data_pkg = default_data_package
        self._data = None # this'll be a dict where we store all our data
        self._src = None
        self._dir = data_dir
        self._name = data_name
        self._key = data_key
        self._src_key = source_key
        self._pkg = data_pkg
        self._alts = alternate_keys
        self._loaded = False
        self.getter = getter
    @property
    def data_file(self): # in case other people want to load it...
        return os.path.join(self._dir, self._name+".py")
    def _load_alts(self):
        # assumes we have dict data, but this entire structure does that anyway
        if not self._alts is None:
            extras = {}
            if isinstance(self._alts, str):
                self._alts = (self._alts,)
            for k in self._alts:
                extras.update({a[k]:a for a in self._data.values()}) #shouldn't increase memory bc mutable
            self._data.update(extras)
    def load(self):
        # currently we only load python data
        # TODO: I should rewrite this to use a Loader object...
        env = {}
        import sys
        sys.path.insert(0, self._dir) #name needs to be unique enough...
        if self._pkg is None:
            exec_stmt = "from {0} import {1} as {1}".format(self._name, self._key)
        else:
            exec_stmt = "from {0}.{1} import {2} as {2}".format(self._pkg, self._name, self._key)
        exec(exec_stmt, env, env)
        self._data = env[self._key]
        try:
            self._src = env[self._src_key]
        except:
            pass
        self._load_alts()
        self._loaded = True
    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data
    @property
    def source(self):
        if self._src is None and not self._loaded:
            self.load()
        return self._src
    def __getitem__(self, key):
        def _get(a, k):
            if k not in a:
                raise DataError("{}: data source {} doesn't have key {}".format(
                    type(self).__name__,
                    self._name,
                    key
                ))
            return a[k]
        data = self.data
        if self.getter is None:
            if isinstance(key, tuple):
                from functools import reduce
                return reduce(_get, key, data)
            else:
                if key not in data:
                    raise DataError("{}: data source {} doesn't have key {}".format(
                        type(self).__name__,
                        self._name,
                        key
                    ))
                return data[key]
        else:
            return self.getter(data, key)
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data.items())