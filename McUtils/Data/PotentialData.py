from .CommonData import DataHandler, DataRecord

__all__ = [ "PotentialData" ]

class PotentialDataHandler(DataHandler):
    def __init__(self):
        super().__init__("PotentialData")
class PotentialDataRecord(DataRecord):
    """
    Represents a simple callable wavefunction...
    """
    def __call__(self, *args, **kwargs):
        return self['Function'](*args, **kwargs)
PotentialData=PotentialDataHandler()
PotentialData.__doc__ = """An instance of PotentialDataHandler that can be used for looking up data on potentials"""
PotentialData.__name__ = "PotentialData"