"""
Provides a class for handling a compiled set of atomic data
"""
from .CommonData import DataHandler

# this might become a singleton class at some point in the future...

class AtomDataHandler(DataHandler):
    def __init__(self):
        super().__init__("AtomData", alternate_keys=("Name", "Symbol", "CanonicalSymbol"))
    def load(self):
        # now update by max IsotopeFraction
        super().load()
        maxIsos = {}
        for v in self._data.values():
            num = v["Number"]
            if num not in maxIsos or v["IsotopeFraction"] > maxIsos[num][0]:
                maxIsos[num] = (v["IsotopeFraction"], v)
        self._data.update(((k, v[1]) for k, v in maxIsos.items()))
AtomData = AtomDataHandler()