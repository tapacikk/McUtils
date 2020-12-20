from Peeves.Doc import *
import os

__pkgs__ = [ "McUtils" ]

root = os.path.dirname(os.path.dirname(__file__))
target = os.path.join(root, "docs")
DocWalker(__pkgs__, target).write_docs()