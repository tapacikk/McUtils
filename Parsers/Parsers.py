"""
A set of concrete parser objects for general use
"""

from .StringParser import *
from .RegexPatterns import *

__all__= [
    "XYZParser"
]

XYZParser = StringParser(
    RegexPattern(
        (
            Named(PositiveInteger, "NumberOfAtoms"),
            Named(Repeating(Any, min = None), "Comment", dtype=str),
            Named(
                Repeating(
                    RegexPattern(
                        (
                            Capturing(AtomName),
                            Duplicated(Capturing(Number), 3, riffle = Whitespace, suffix=Optional(Whitespace))
                        ),
                        joiner=Whitespace
                    ),
                    suffix=Optional(Newline)
                ),
                "Atoms"
            )
        ),
        "XYZ",
        joiner=Newline
    )
)


# there's a subtle difference between Duplicated and Repeating
# Duplicated copies the pattern directly a set number of times which allows it to
# capture every single instance of the pattern
# Repeating uses Regex syntax to repeat a pattern a potentially unspecified number of times
# which means the parser will only return the first case when asked for the groups