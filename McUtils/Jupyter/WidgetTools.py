
__all__ = [
    "JupyterAPIs"
]

class JupyterAPIs:
    """
    Provides access to the various Jupyter APIs
    """

    _apis = None
    @classmethod
    def load_api(cls):
        try:
            import IPython.core.display as display
        except ImportError:
            display = None
        try:
            import ipywidgets as widgets
        except ImportError:
            widgets = None
        try:
            import ipyevents as events
        except ImportError:
            events = None

        cls._apis = (
            display,
            widgets,
            events
        )

    @classmethod
    def get_display_api(cls):
        if cls._apis is None:
            cls.load_api()
        return cls._apis[0]

    @property
    def display_api(self):
        return self.get_display_api()

    @classmethod
    def get_widgets_api(self):
        if self._apis is None:
            self.load_api()
        return self._apis[1]

    @property
    def widgets_api(self):
        return self.get_widgets_api()

    @classmethod
    def get_events_api(self):
        if self._apis is None:
            self.load_api()
        return self._apis[2]

    @property
    def events_api(self):
        return self.get_events_api()