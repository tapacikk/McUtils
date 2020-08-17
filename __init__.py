"""
a growing package of assorted functionality that finds use across many different packages, but doesn't attempt to
provide a single unified interface for doing certain types of projects

All of the McUtils packages stand mostly on their own, but there will be little calls into one another here and there
"""

import McUtils.Data
import McUtils.GaussianInterface
import McUtils.Misc
import McUtils.Parsers
import McUtils.Plots
import McUtils.Zachary
import McUtils.Coordinerds
import McUtils.ExternalPrograms

__all__ = [
    "Data",
    "GaussianInterface",
    "Misc",
    "Parsers",
    "Plots",
    "Zachary",
    "Coordinerds",
    "ExternalPrograms"
]