from .CommonData import DataHandler

__all__ = [ "PotentialData" ]

class PotentialDataHandler(DataHandler):
    def __init__(self):
        super().__init__("PotentialData")
PotentialData=PotentialDataHandler()
PotentialData.__doc__ = """An instance of PotentialDataHandler that can be used for looking up data on potentials"""
PotentialData.__name__ = "PotentialData"