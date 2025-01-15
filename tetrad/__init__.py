#!/usr/bin/env python

"""Tetrad quartet supertree inference tool.

Change log
==========

1.0
---
- dropped support for Python<3.7
- require ipyparallel>=7.0
- require toytree>3.0
- require typer

TODO
-----
- insert a test here to check MKL and OPENBLAS...
- clone vcf convert from ipyrad
- add linkage option to CLI
- speed up converter using numba
"""

__version__ = "1.0.0"
__author__ = "Deren Eaton"


# from tetrad.core import Tetrad
# from tetrad.logger_setup import set_log_level
# set_log_level("INFO")
