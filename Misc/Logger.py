import os, enum

__all__ = [
    "Logger",
    "LogLevel",
    "NullLogger"
]


class LogLevel(enum.Enum):
    """
    A simple log level object to standardize more pieces of the logger interface
    """
    Quiet = 0
    Warnings = 1
    Debug = 10
    All = 100

    def __eq__(self, other):
        if isinstance(other, LogLevel):
            other = other.value
        return self.value == other
    def __le__(self, other):
        if isinstance(other, LogLevel):
            other = other.value
        return self.value <= other
    def __ge__(self, other):
        if isinstance(other, LogLevel):
            other = other.value
        return self.value >= other
    def __lt__(self, other):
        if isinstance(other, LogLevel):
            other = other.value
        return self.value < other
    def __gt__(self, other):
        if isinstance(other, LogLevel):
            other = other.value
        return self.value > other

class Logger:
    """
    Defines a simple logger object to write log data to a file based on log levels.
    """

    def __init__(self, log_file = None, verbosity = LogLevel.All, padding="", newline="\n"):
        self.log_file = log_file
        self.verbosity = verbosity
        self.padding = padding
        self.newline = newline

    def format_message(self, message, *params, **kwargs):
        if len(kwargs) > 0:
            message = message.format(**kwargs)
        elif len(params) > 0:
            message = message.format(*params)
        return message

    def log_print(self, message, *params, print_options=None, padding=None, newline=None, **kwargs):
        """
        :param message: message to print
        :type message: str | Iterable[str]
        :param params:
        :type params:
        :param print_options: options to be passed through to print
        :type print_options:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if padding is None:
            padding = self.padding
        if newline is None:
            newline = self.newline

        if not isinstance(message, str):
            joiner = (newline + padding)
            message = joiner.join(
                [padding + message[0]]
                + list(message[1:])
            )
        else:
            message = padding + message

        # print(">>>>", repr(message), params)

        if print_options is None:
            print_options={}
        if 'verbosity' in kwargs:
            verbosity = kwargs['verbosity']
            del kwargs['verbosity']
        else:
            verbosity = 0

        if verbosity <= self.verbosity:
            log = self.log_file
            if isinstance(log, str):
                if not os.path.isdir(os.path.dirname(log)):
                    try:
                        os.makedirs(os.path.dirname(log))
                    except OSError:
                        pass
                #O_NONBLOCK is *nix only
                with open(log, "a", os.O_NONBLOCK) as lf: # this is potentially quite slow but I am also quite lazy
                    print(self.format_message(message, *params, **kwargs), file=lf, **print_options)
            elif log is None:
                print(self.format_message(message, *params, **kwargs), **print_options)
            else:
                print(self.format_message(message, *params, **kwargs), file=log, **print_options)

class NullLogger(Logger):
    """
    A logger that implements the interface, but doesn't ever print.
    Allows code to avoid a bunch of "if logger is not None" blocks
    """
    def log_print(self, message, *params, print_options=None, padding=None, newline=None, **kwargs):
        pass