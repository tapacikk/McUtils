"""Defines common data handler

"""
import os, sys

default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
default_data_package = "TheRealMcCoy"
default_data_key = "data"
class DataHandler:
    """Defines a general data loader class that we can use for AtomData and any other data classes we might find useful

    """
    def __init__(self,
                 data_name,
                 data_key = default_data_key,
                 data_dir = default_data_dir,
                 data_pkg = default_data_package,
                 alternate_keys = None
                 ):
        self._data = None # this'll be a dict where we store all our data
        self._dir = data_dir
        self._name = data_name
        self._key = data_key
        self._pkg = data_pkg,
        self._alts = alternate_keys
    @property
    def data_file(self): # in case other people want to load it...
        return os.path.join(self._dir, self._name+".py")
    def _load_alts(self):
        # assumes we have dict data, but this entire structure does that anyway
        if not self._alts is None:
            extras = {}
            for k in self._alts:
                extras.update({a[k]:a for a in self._data.items()}) #shouldn't increase memory bc mutable
            self._data.update(extras)
    def load(self):
        # currently we only load python data
        env = {}
        import sys
        sys.path.insert(0, ) #name needs to be unique enough...
        if self._pkg is None:
            exec_stmt = "import " + self._name
        else:
            exec_stmt = "import {}.{}".format(self._pkg, self._name)
        exec(exec_stmt, env, env)
        self._data = env[self._key]
        self._load_alts()
    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data
    def __getitem__(self, key):
        return self.data[key]
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data.items())