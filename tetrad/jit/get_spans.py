#!/usr/bin/env python

"""

"""

import numpy as np
from numba import njit


@njit
def jit_get_spans(maparr: np.ndarray) -> np.ndarray:
    """Return an array with span distances for each locus.

    This is much faster than pandas or numpy queries.
    [ 0, 33],
    [33, 47],
    [47, 51], ...
    """
    sidx = maparr[0, 0]
    locs = np.unique(maparr[:, 0])
    nlocs = locs.size
    spans = np.zeros((nlocs, 2), np.int64)

    lidx = 0
    # advance over all snp rows 
    for idx in range(maparr.shape[0]):

        # get locus id at this row 0, 0, 0, 0
        eidx = maparr[idx, 0]

        # if locus id is not sidx
        if eidx != sidx:

            # the first value entered
            if not lidx:
                spans[lidx] = np.array((0, idx))

            # all other values
            else:
                spans[lidx] = spans[lidx - 1, 1], idx

            lidx += 1
            sidx = locs[lidx]

    # final end span
    spans[-1] = np.array((spans[-2, 1], maparr[-1, -1] + 1))
    return spans


if __name__ == "__main__":

    MAP = "..."
    SPANS = jit_get_spans(MAP)
    print(SPANS)
