from .CommonData import DataHandler

__all__ = [ "WavefunctionData" ]

class WavefunctionDataHandler(DataHandler):
    def __init__(self):
        super().__init__("WavefunctionData")
WavefunctionData=WavefunctionDataHandler()
WavefunctionData.__doc__ = """An instance of WavefunctionDataHandler that can be used for looking up data on wavefunctions"""
WavefunctionData.__name__ = "WavefunctionData"