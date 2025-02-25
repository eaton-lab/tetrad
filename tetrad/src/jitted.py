#!/usr/bin/env python

"""Jitted funcs.

"""

import numpy as np
from numba import njit


GETCONS = np.array([
    [82, 71, 65],
    [75, 71, 84],
    [83, 71, 67],
    [89, 84, 67],
    [87, 84, 65],
    [77, 67, 65]], dtype=np.uint8,
)


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
def jit_resolve_ambigs(tmpseq: np.ndarray, seed: int) -> np.ndarray:
    """Randomly resolve ambiguous bases in a uint8 seqarray. 

    This is applied to each boot replicate so that over reps the 
    random resolutions don't matter. Sites are randomly resolved, 
    so best for unlinked SNPs since otherwise linked SNPs are losing 
    their linkage information... though it's not like we're using
    it anyways.
    """
    np.random.seed(seed)

    # iter rows in GETCONS (np.uint([82, 75, 83, 89, 87, 77]))
    for aidx in range(6):
        ambig, res1, res2 = GETCONS[aidx]

        # get wher tmpseq == ambig and assign half to each resolution
        idx, idy = np.where(tmpseq == ambig)
        halfmask = np.random.binomial(n=1, p=0.5, size=idx.shape[0])
        for col in range(idx.shape[0]):
            if halfmask[col]:
                tmpseq[idx[col], idy[col]] = res1
            else:
                tmpseq[idx[col], idy[col]] = res2
    return tmpseq


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