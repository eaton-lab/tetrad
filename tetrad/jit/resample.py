#!/usr/bin/env python

import numpy as np
from numba import njit


@njit
def jit_get_nsites(spans: np.ndarray, lidxs: np.ndarray) -> int:
    """Return size of resampled array in nsites.

    The shape can change because when we resample loci the 
    number of SNPs in the data set can change.
    """
    width = 0
    for idx in range(lidxs.size):
        width += spans[lidxs[idx], 1] - spans[lidxs[idx], 0]
    return width


@njit
def jit_resample(
    seqarr: np.ndarray, 
    spans: np.ndarray, 
    lidxs: np.ndarray, 
    seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Fill the tmparr SNP array with SNPs from resampled loci."""

    np.random.seed(seed)
    arrlen = jit_get_nsites(spans, lidxs)
    tmparr = np.zeros((seqarr.shape[0], arrlen), dtype=np.uint8)
    tmpmap = np.zeros((arrlen, 2), dtype=np.uint32)
    tmpmap[:, 1] = np.arange(arrlen, dtype=np.uint32)

    # current column index as we move from left to right filling tmparr
    cidx = 0

    # iterate over resampled locus indices  
    # (0, 20), (1, 400), (2, 33), (3, 666), (4, 20), ...
    for idx, lidx in enumerate(lidxs):
        
        # spans are (start,end) positions of lidx in seqarr.
        start, end = spans[lidx]

        # getch locus lidx from seqarr using the span indices.
        cols = seqarr[:, start:end]

        # randomize order of columns within the locus
        col_idxs = np.random.choice(cols.shape[1], cols.shape[1], replace=False)
        randomized_cols = cols[:, col_idxs]
        
        # NOTE: the required length was already measured
        # fill tmparr with n columns from seqarr
        tmparr[:, cidx:cidx + cols.shape[1]] = randomized_cols

        # fill tmpmap with new map info from these resampled loci
        # 0-46 -> 
        tmpmap[cidx: cidx + cols.shape[1], 0] = idx
        
        # advance column index
        cidx += cols.shape[1]

    # return the concatenated cols
    return tmparr, tmpmap
