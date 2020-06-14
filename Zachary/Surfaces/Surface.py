class Surface:
    """
    This actually isn't a concrete implementation of BaseSurface.
    Instead it's a class that _dispatches_ to an implementation of BaseSurface to do its core evaluations (plus it does shape checking)
    """
    def __init__(self, data, dimension=None, base=None, **metadata):
        if base is None:
            base = self.detect_base(data)
        self.metadata = metadata
        self.base = base(*data, dimension)

    @classmethod
    def detect_base(cls, data):
        ...
    def __call__(self, gridpoints, **kwargs):
        return self.base(gridpoints, **kwargs)
