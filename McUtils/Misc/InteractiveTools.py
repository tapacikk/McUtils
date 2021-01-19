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

        base = self.mod.__all__ if hasattr(self.mod, '__all__') else dir(self.mod)
        if hasattr(self.mod, '__reload_hook__'):
            try:
                others = list(self.mod.__reload_hook__)
            except TypeError:
                pass
            else:
                base = list(base) + others
        return base

    blacklist_keys = ['site-packages', os.path.abspath(os.path.dirname(inspect.getfile(os)))]
    def reload(self, 
        stack=None,
        reloaded=None, blacklist=None, reload_parents=True, 
        verbose=False,
        print_indent=""):
        """
        Recursively searches for modules to reload and then reloads them.
        Uses a cache to break cyclic dependencies of any sort.
        This turns out to also be a challenging problem, since we need to basically
        load depth-first, while never jumping too far back...


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
            if verbose:
                print(print_indent + "Reloading:", self.mod.__name__)
            reloaded.add(self.mod.__name__)

            print_indent += "  "

            for member in self.get_members():
                # print(print_indent + " member:", member)
                if member.startswith('.'):
                    how_many = 0
                    while member[how_many] == ".":
                        how_many += 1
                        if how_many == len(member):
                            break
                    main_name = self.mod.__name__.rsplit(".", how_many)[0]
                    test_key = main_name + "." + member[how_many:]
                else:
                    test_key = self.mod.__name__ + "." + member
                if test_key in sys.modules:
                    type(self)(test_key).reload(
                        reloaded=reloaded, blacklist=blacklist, 
                        verbose=verbose,
                        reload_parents=reload_parents, print_indent=print_indent
                        )
                else:
                    obj = getattr(self.mod, member)
                    if isinstance(obj, types.ModuleType):
                        type(self)(obj).reload(
                            reloaded=reloaded, blacklist=blacklist, 
                            verbose=verbose,
                            reload_parents=reload_parents, print_indent=print_indent
                            )
                    elif isinstance(obj, type):
                        type(self)(obj.__module__).reload(
                            reloaded=reloaded, blacklist=blacklist, 
                            verbose=verbose,
                            reload_parents=reload_parents, print_indent=print_indent
                            )
                    else:
                        obj = type(obj)
                        type(self)(obj.__module__).reload(
                            reloaded=reloaded, blacklist=blacklist, 
                            verbose=verbose,
                            reload_parents=reload_parents, print_indent=print_indent
                            )
           
            
            if hasattr(self.mod, '__reload_hook__'):
                try:
                    self.mod.__reload_hook__()
                except TypeError:
                    pass
            if verbose:
                print(print_indent + "loading:", self.mod.__name__)
            importlib.reload(self.mod)


            load_parents = []
            if reload_parents:
                # make sure parents get loaded in the appropriate order...
                for parent in self.get_parents():
                    if parent in reloaded:
                        # prevent us from jumping back too far...
                        break
                    # print(" storing", parent)
                    load_parents.append(parent)
                    type(self)(parent).reload(
                        reloaded=reloaded, blacklist=blacklist, 
                        reload_parents=reload_parents, verbose=verbose,
                        print_indent=print_indent
                        )
