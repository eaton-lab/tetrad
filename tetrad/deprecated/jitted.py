#!/usr/bin/env python

"""Just in time compiled functions for tetrad

"""

from typing import Tuple
import numpy as np
from numba import njit
from tetrad.utils import GETCONS


# @njit()
# def calculate(seqnon, mapcol, nmask, tests):
#     """Groups together several other numba funcs.
#     """
#     # get the invariants matrix
#     mats = chunk_to_matrices(seqnon, mapcol, nmask)

#     # empty arrs to fill
#     svds = np.zeros((3, 16), dtype=np.float64)
#     scor = np.zeros(3, dtype=np.float64)
#     rank = np.zeros(3, dtype=np.float64)

#     # svd and rank.
#     for test in range(3):
#         svds[test] = np.linalg.svd(mats[test].astype(np.float64))[1]
#         rank[test] = np.linalg.matrix_rank(mats[test].astype(np.float64))

#     # get minrank, or 11 (TODO: can apply seq model here)
#     minrank = int(min(10, rank.min()))
#     for test in range(3):
#         scor[test] = np.sqrt(np.sum(svds[test, minrank:]**2))

#     # sort to find the best qorder
#     best = np.where(scor == scor.min())[0]
#     bidx = tests[best][0]

#     # returns the best quartet resolution and the first matrix (for invars)
#     return bidx, mats[0]


# @njit('u4[:,:,:](u1[:,:],u4[:],b1[:])')
# def chunk_to_matrices(narr, mapcol, nmask):
#     """numba compiled code to get matrix fast.

#     arr is a 4 x N seq matrix converted to np.int8
#     I convert the numbers for ATGC into their respective index for the MAT
#     matrix, and leave all others as high numbers, i.e., -==45, N==78.        
#     """

#     ## get seq alignment and create an empty array for filling
#     mats = np.zeros((3, 16, 16), dtype=np.uint32)

#     ## replace ints with small ints that index their place in the 
#     ## 16x16. This no longer checks for big ints to exclude, so resolve=True
#     ## is now the default, TODO. 
#     last_loc = -1
#     for idx in range(mapcol.shape[0]):
#         if not nmask[idx]:
#             if not mapcol[idx] == last_loc:
#                 i = narr[:, idx]
#                 mats[0, (4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1      
#                 last_loc = mapcol[idx]

#     ## fill the alternates
#     x = np.uint8(0)
#     for y in np.array([0, 4, 8, 12], dtype=np.uint8):
#         for z in np.array([0, 4, 8, 12], dtype=np.uint8):
#             mats[1, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4)
#             mats[2, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4).T
#             x += np.uint8(1)
#     return mats

# @njit
# def old_jget_spans(maparr, spans):
#     """Get span distance for each locus in original seqarray. 

#     This is used to create re-sampled arrays in each bootstrap 
#     to sample unlinked SNPs. 
#     """
#     ## start at 0, finds change at 1-index of map file
#     bidx = 1
#     spans = np.zeros((maparr[-1, 0], 2), np.uint64)
#     ## read through marr and record when locus id changes
#     for idx in range(1, maparr.shape[0]):
#         cur = maparr[idx, 0]
#         if cur != bidx:
#             # idy = idx + 1
#             spans[cur - 2, 1] = idx
#             spans[cur - 1, 0] = idx
#             bidx = cur
#     spans[-1, 1] = maparr[-1, -1]
#     return spans


# @njit()
# def jshuffle_cols(seqarr, newarr, cols):
#     "Used in bootstrap resampling when no map file is present."
#     for idx in range(cols.shape[0]):
#         newarr[:, idx] = seqarr[:, cols[idx]]
#     return newarr


##################################################################

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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


# @njit(parallel=True)
# def subsample_loci_and_snps(
#     snpsmap: np.ndarray, 
#     resample_loci: bool,
#     rng: np.random.Generator,
#     ) -> np.ndarray:
#     """Subsample loci and SNPs (one per locus) using snpsmap

#     """
#     # get locus idxs
#     sidxs = np.unique(snpsmap[:, 0])

#     # resample locus idxs
#     if resample_loci:
#         sidxs = rng.choice(sidxs, sidxs.size)

#     # an array to fill
#     subs = np.zeros(sidxs.size, dtype=np.int64)
#     idx = 0

#     # iterate over loci 
#     for sidx in sidxs:

#         # get all sites in this locus
#         sites = snpsmap[snpsmap[:, 0] == sidx, 1]

#         # randomly select one and save
#         site = rng.choice(sites)
#         subs[idx] = site
#         idx += 1

#     # return as sorted array of int64
#     subs.sort()
#     return subs



# deprecated: much slower than chunk_to_matrices.
@njit()
def calc2(seqs, maparr, mapmask, tests):

    # get snpsmap masked for this quartet set
    snpsmap = maparr[mapmask, :]

    # empty for storing output 
    mats = np.zeros((3, 16, 16), dtype=np.uint32)

    # get all loci that still contain a snp
    loci = np.unique(snpsmap[:, 0])

    # iterate over loci sampling one SNP from each
    idx = 0
    for loc in loci:
        sidxs = snpsmap[snpsmap[:, 0] == loc, 1]
        sidx = np.random.choice(sidxs)
        i = seqs[:, sidx]

        # enter the site into the counts matrix
        mats[0, (4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1      
        idx += 1

    # fill the alternates
    x = np.uint8(0)
    qidxs = np.array([0, 4, 8, 12], dtype=np.uint8)
    for y in qidxs:
        for z in qidxs:

            # mat x
            remat = mats[0, x].reshape(4, 4)
            mats[1, y:y + np.uint8(4), z:z + np.uint8(4)] = remat

            # mat y
            remat = mats[0, x].reshape(4, 4).T
            mats[2, y:y + np.uint8(4), z:z + np.uint8(4)] = remat

            # increment counter
            x += np.uint8(1)

    # empty arrs to fill
    svds = np.zeros((3, 16), dtype=np.float64)
    scor = np.zeros(3, dtype=np.float64)
    rank = np.zeros(3, dtype=np.float64)

    # svd and rank.
    for test in range(3):
        svds[test] = np.linalg.svd(mats[test].astype(np.float64))[1]
        rank[test] = np.linalg.matrix_rank(mats[test].astype(np.float64))

    # get minrank, or 11 (TODO: can apply seq model here)
    minrank = int(min(11, rank.min()))
    for test in range(3):
        scor[test] = np.sqrt(np.sum(svds[test, minrank:]**2))

    # sort to find the best qorder
    best = np.where(scor == scor.min())[0]
    bidx = tests[best][0]

    return bidx, mats[0]


if __name__ == "__main__":

    arr = np.column_stack([
        np.concatenate([np.repeat(0, 10), np.repeat(1, 10)]),
        np.arange(20),
    ])
    print(jit_get_spans(arr))
