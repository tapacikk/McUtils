import abc
from collections import OrderedDict

__all__ = [
    "Cache",
    "MaxSizeCache"
]

class Cache(metaclass=abc.ABCMeta):
    """
    Simple cache base class
    """
    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError("Cache is an abstract base class")
    @abc.abstractmethod
    def __contains__(self, item):
        raise NotImplementedError("Cache is an abstract base class")
    @abc.abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError("Cache is an abstract base class")

class MaxSizeCache:
    """
    Simple lru-cache to support ravel/unravel ops
    """
    def __init__(self, max_items=128):
        self.od = OrderedDict()
        self.max_items = max_items
    def __contains__(self, item):
        return item in self.od
    def __getitem__(self, item):
        val = self.od[item]
        self.od.move_to_end(item)
        return val
    def __setitem__(self, key, value):
        self.od[key] = value
        self.od.move_to_end(key)
        if len(self.od) > self.max_items:
            self.od.popitem(last=False)