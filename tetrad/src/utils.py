#!/usr/bin/env python

"""Utilities for tetrad.

"""

import sys
import subprocess
from pathlib import Path
import numpy as np


# used by resolve_ambigs
GETCONS = np.array([
    [82, 71, 65],
    [75, 71, 84],
    [83, 71, 67],
    [89, 84, 67],
    [87, 84, 65],
    [77, 67, 65]], dtype=np.uint8,
)


def make_wide(formatter, w=120, h=36):
    """Return a wider HelpFormatter, if possible."""
    try:
        # https://stackoverflow.com/a/5464440
        # beware: "Only the name of this class is considered a public API."
        kwargs = {'width': w, 'max_help_position': h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        return formatter


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
            raise IOError(f"No QMC binary found: {qmc}")
    return qmc


if __name__ == "__main__":
    print(get_qmc_binary())
