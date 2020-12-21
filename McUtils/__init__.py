"""
A growing package of assorted functionality that finds use across many different packages, but doesn't attempt to
provide a single unified interface for doing certain types of projects.

All of the McUtils packages stand mostly on their own, but there will be little calls into one another here and there.

The more scientifically-focused `Psience` package makes significant use of `McUtils`.
"""

import McUtils.Data
import McUtils.GaussianInterface
import McUtils.Misc
import McUtils.Parsers
import McUtils.Plots
import McUtils.Zachary
import McUtils.Coordinerds
import McUtils.ExternalPrograms
import McUtils.Extensions

__all__ = [
    "Data",
    "GaussianInterface",
    "Misc",
    "Parsers",
    "Plots",
    "Zachary",
    "Coordinerds",
    "ExternalPrograms",
    "Extensions",
    "Numputils",
    "Scaffolding"
]