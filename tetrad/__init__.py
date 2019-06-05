#!/usr/bin/env python

import os as _os
import sys as _sys
from tetrad.tetrad import Tetrad as tetrad


# dunders mifflin
__version__ = "0.8.0"
__author__ = "Deren Eaton"
__interactive__ = 1


# path to here
_tet_path = _os.path.dirname(_os.path.dirname(
    _os.path.abspath(_os.path.dirname(__file__))))

# path to binaries relative to here
_bin_path = _os.path.join(_tet_path, "bin")

# path to QMC binary from bin 
qmc = (
    _os.path.join(_os.path.abspath(_bin_path), "QMC-linux-x86_64")
    if "linux" in _sys.platform else 
    _os.path.join(_os.path.abspath(_bin_path), "QMC-osx-x86_64") 
)
