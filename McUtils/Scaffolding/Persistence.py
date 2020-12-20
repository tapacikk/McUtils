"""
Provides utilities for managing object persistence.
Doesn't focus particularly heavily on serialization/deserialization, since
the Serializers package does that just fine.
More interested in managing objects that provide the ability to load from a configuration
and which can be stored in a directory or ZIP file.
"""

import os, shutil

from .Configurations import Config

__all__ = [
    "PersistenceManager"
]

class PersistenceManager:
    """
    Defines a manager that can load configuration data from a directory
    or, maybe in the future, a SQL database or similar.
    Requires class that supports `from_config` to load and `to_config` to save.
    """
    def __init__(self, cls, persistence_loc = None):
        """
        :param cls:
        :type cls: type
        :param persistence_loc: location from which to load/save objects
        :type persistence_loc: str | None
        """
        self.cls = cls
        if persistence_loc is None:
            if not hasattr(cls, 'persistence_loc'):
                raise AttributeError("{me}: to support persistence either a '{ploc}'' must be passed or {cls} needs the attribute '{ploc}'".format(
                    me=type(self).__name__,
                    cls=cls.__name__,
                    ploc=persistence_loc
                ))
            persistence_loc = cls.persistence_loc
        self.loc = persistence_loc

    def obj_loc(self, key):
        return os.path.join(self.loc, key)

    def load_config(self, key, make_new=False, init=None):
        """
        Loads the config for the persistent structure named `key`
        :param key:
        :type key:
        :return:
        :rtype:
        """
        if self.contains(key):
            return Config(self.obj_loc(key))
        elif make_new:
            return self.new_config(key, init=init)
        else:
            raise KeyError("{}: no persistent object {}".format(
                type(self).__name__,
                key
            ))

    def new_config(self, key, init=None):
        """
        Creates a new space and config for the persistent structure named `key`
        :param key:
        :type key:
        :return:
        :rtype:
        """
        loc = self.obj_loc(key)
        if not os.path.isdir(loc):
            os.makedirs(loc, exist_ok=True)
        return Config.new(loc, init=init)

    def contains(self, key):
        """
        Checks if `key` is a supported persistent structure

        :param key:
        :type key:
        :return:
        :rtype:
        """
        return os.path.isdir(self.obj_loc(key))

    def load(self, key, make_new=False, strict=True, init=None):
        """
        Loads the persistent structure named `key`

        :param key:
        :type key:
        :return:
        :rtype:
        """
        cfg = self.load_config(key, make_new=make_new, init=init)
        try:
            loader = self.cls.from_config
        except AttributeError:
            loader = None
        if loader is None:
            raise AttributeError("{}.{}: to support persistence {} has to have a classmethod '{}'".format(
                type(self).__name__,
                'load',
                self.cls,
                'from_config'
            ))
        return cfg.apply(loader, strict=strict)
    def save(self, obj):
        """
        Saves requisite config data for a structure

        :param obj:
        :type obj:
        :return:
        :rtype:
        """
        if not isinstance(obj, self.cls):
            raise TypeError("{}.{}: object {} isn't of persistence type {}".format(
                type(self).__name__,
                'save',
                obj,
                self.cls.__name__
            ))

        try:
            loader = obj.to_config
        except AttributeError:
            loader = None
        if loader is None:
            raise AttributeError("{}.{}: to support persistence {} has to have a classmethod '{}'".format(
                type(self).__name__,
                'save',
                self.cls,
                'to_config'
            ))
        data = loader()

        key = data['name']
        cfg = self.load_config(key, make_new=True)
        cfg.update(**data)

