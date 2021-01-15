"""
Miscellaneous tools for interactive messing around in Jupyter environments
"""
import sys, os, types, importlib, inspect

__all__ = [
    "ModuleReloader"
]

class ModuleReloader:
    """
    Reloads a module & recursively descends its 'all' tree
    to make sure that all submodules are also reloaded
    """

    def __init__(self, modspec):
        """
        :param modspec:
        :type modspec: str | types.ModuleType
        """
        if isinstance(modspec, str):
            modspec=sys.modules[modspec]
        self.mod = modspec

    def get_parents(self):
        """
        Returns module parents
        :return:
        :rtype:
        """
        split = self.mod.__name__.split(".")
        return [".".join(split[:i]) for i in range(len(split)-1, 0, -1)]

    def get_members(self):
        """
        Returns module members
        :return:
        :rtype:
        """
        return self.mod.__all__ if hasattr(self.mod, '__all__') else dir(self.mod)

    blacklist_keys = ['site-packages', os.path.abspath(os.path.dirname(inspect.getfile(os)))]
    def reload(self, reloaded=None, blacklist=None, reload_parents=True):
        """
        Recursively searches for modules to reload and then reloads them.
        Uses a cache to break cyclic dependencies of any sort

        :return:
        :rtype:
        """

        if reloaded is None:
            reloaded = set()

        if blacklist is None:
            blacklist = set()
        blacklist.update(sys.builtin_module_names)

        key = self.mod.__name__
        if (
                key not in reloaded
                and key not in blacklist
                and all(k not in inspect.getfile(self.mod) for k in self.blacklist_keys)
        ):
            reloaded.add(self.mod.__name__)
            for member in self.get_members():
                test_key = self.mod.__name__ + "." + member
                if test_key in sys.modules:
                    type(self)(test_key).reload(reloaded=reloaded, blacklist=blacklist, reload_parents=reload_parents)
                else:
                    obj = getattr(self.mod, member)
                    if isinstance(obj, types.ModuleType):
                        type(self)(obj).reload(reloaded=reloaded, blacklist=blacklist, reload_parents=reload_parents)
                    elif isinstance(obj, type):
                        type(self)(obj.__module__).reload(reloaded=reloaded, blacklist=blacklist, reload_parents=reload_parents)
            # print(self.mod.__name__)
            importlib.reload(self.mod)
            if reload_parents:
                for parent in self.get_parents():
                    # check to make sure that we're not needlessly walking back up the tree
                    if parent in reloaded:
                        break
                    type(self)(parent).reload(reloaded=reloaded, blacklist=blacklist, reload_parents=reload_parents)


