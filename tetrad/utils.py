#!/usr/bin/env python

"""Utilities for tetrad.

Helper classes for tetrad based on similar utils in ipyrad"

Logger for development and user warnings.

All toytree modules that use logging use a 'bound' logger that will
be filtered here to only show for toytree and not for other Python
packages.
"""

import sys
import subprocess
from pathlib import Path
import numpy as np

class TetradError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

# used by resolve_ambigs
GETCONS = np.array([
    [82, 71, 65],
    [75, 71, 84],
    [83, 71, 67],
    [89, 84, 67],
    [87, 84, 65],
    [77, 67, 65]], dtype=np.uint8,
)


def get_qmc_binary() -> Path:
    """Find and return the QMC binary path."""
    # If conda installed then the QMC binary should be in this conda env bin 
    platform = "Linux" if "linux" in sys.platform else "Mac"
    binary = f"find-cut-{platform}-64"
    qmc = Path(sys.prefix) / "bin" / Path(binary)

    # if pip+github installed then QMC will be relative to this file
    if not qmc.exists():
        tetpath = Path(__file__).absolute().parent.parent
        binpath = tetpath / "bin"
        qmc = binpath.absolute() / binary

    # check for binary
    with subprocess.Popen(
        ["which", qmc], stderr=subprocess.STDOUT, stdout=subprocess.PIPE
        ) as proc:
        res = proc.communicate()[0]
        if not res:
            raise TetradError(f"No QMC binary found: {qmc}")
    return qmc

if __name__ == "__main__":
    print(get_qmc_binary())
