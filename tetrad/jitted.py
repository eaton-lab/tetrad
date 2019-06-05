#!/usr/bin/env python

"""
Just in time compiled functions for tetrad
"""

import numpy as np
from numba import njit, prange


# used by resolve_ambigs
GETCONS = np.array([
    [82, 71, 65],
    [75, 71, 84],
    [83, 71, 67],
    [89, 84, 67],
    [87, 84, 65],
    [77, 67, 65]], dtype=np.uint8,
)


njit(parallel=True)
def get_spans(maparr, spans):
    """ 
    Get span distance for each locus in original seqarray. This
    is used to create re-sampled arrays in each bootstrap to sample
    unlinked SNPs from. Used on snpsphy or str or ...
    """
    start = 0
    end = 0
    for idx in prange(1, spans.shape[0] + 1):
        lines = maparr[maparr[:, 0] == idx]
        if lines.size:
            end = lines[:, 3].max()
            spans[idx - 1] = [start, end]
        else: 
            spans[idx - 1] = [end, end]
        start = spans[idx - 1, 1]

    # drop rows with no span (invariant loci)
    spans = spans[spans[:, 0] != spans[:, 1]]
    return spans


@njit()
def calculate(seqnon, mapcol, nmask, tests):
    """
    Groups together several other numba funcs.
    """
    ## get the invariants matrix
    mats = chunk_to_matrices(seqnon, mapcol, nmask)

    ## empty arrs to fill
    svds = np.zeros((3, 16), dtype=np.float64)
    scor = np.zeros(3, dtype=np.float64)
    rank = np.zeros(3, dtype=np.float64)

    ## why svd and rank?
    for test in range(3):
        svds[test] = np.linalg.svd(mats[test].astype(np.float64))[1]
        rank[test] = np.linalg.matrix_rank(mats[test].astype(np.float64))

    ## get minrank, or 11
    minrank = int(min(11, rank.min()))
    for test in range(3):
        scor[test] = np.sqrt(np.sum(svds[test, minrank:]**2))

    ## sort to find the best qorder
    best = np.where(scor == scor.min())[0]
    bidx = tests[best][0]

    return bidx, mats[0]



@njit('u4[:,:,:](u1[:,:],u4[:],b1[:])')
def chunk_to_matrices(narr, mapcol, nmask):
    """ 
    numba compiled code to get matrix fast.
    arr is a 4 x N seq matrix converted to np.int8
    I convert the numbers for ATGC into their respective index for the MAT
    matrix, and leave all others as high numbers, i.e., -==45, N==78. 
    """

    ## get seq alignment and create an empty array for filling
    mats = np.zeros((3, 16, 16), dtype=np.uint32)

    ## replace ints with small ints that index their place in the 
    ## 16x16. This no longer checks for big ints to exclude, so resolve=True
    ## is now the default, TODO. 
    last_loc = -1
    for idx in range(mapcol.shape[0]):
        if not nmask[idx]:
            if not mapcol[idx] == last_loc:
                i = narr[:, idx]
                mats[0, (4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1      
                last_loc = mapcol[idx]

    ## fill the alternates
    x = np.uint8(0)
    for y in np.array([0, 4, 8, 12], dtype=np.uint8):
        for z in np.array([0, 4, 8, 12], dtype=np.uint8):
            mats[1, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4)
            mats[2, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4).T
            x += np.uint8(1)

    return mats



@njit()
def jshuffle_cols(seqarr, newarr, cols):
    "Used in bootstrap resampling when no map file is present."
    for idx in range(cols.shape[0]):
        newarr[:, idx] = seqarr[:, cols[idx]]
    return newarr


@njit()
def jget_shape(spans, loci):
    """ 
    Returns shape of new bootstrap resampled locus array. The 
    shape can change because when we resample loci the number
    of SNPs in the data set can change.
    """
    width = 0
    for idx in range(loci.size):
        width += spans[loci[idx], 1] - spans[loci[idx], 0]
    return width


@njit()
def jfill_boot(seqarr, newboot, newmap, spans, loci):
    """ 
    Fills the new bootstrap SNP array and map array with
    new data based on the resampled loci for this boot.
    """
    ## column index
    cidx = 0
  
    ## resample each locus
    for i in range(loci.shape[0]):
        
        ## grab a random locus's columns
        x1 = spans[loci[i]][0]
        x2 = spans[loci[i]][1]
        cols = seqarr[:, x1:x2]

        ## randomize columns within colsq
        cord = np.random.choice(cols.shape[1], cols.shape[1], replace=False)
        rcols = cols[:, cord]
        
        ## fill bootarr with n columns from seqarr
        ## the required length was already measured
        newboot[:, cidx:cidx + cols.shape[1]] = rcols

        ## fill bootmap with new map info
        newmap[cidx: cidx + cols.shape[1], 0] = i + 1
        
        ## advance column index
        cidx += cols.shape[1]

    ## return the concatenated cols
    return newboot, newmap



@njit()
def resolve_ambigs(tmpseq):
    """ 
    Randomly resolve ambiguous bases. This is applied to each boot
    replicate so that over reps the random resolutions don't matter.
    Sites are randomly resolved, so best for unlinked SNPs since 
    otherwise linked SNPs are losing their linkage information... 
    though it's not like we're using it anyways.
    """

    ## the order of rows in GETCONS
    for aidx in range(6):
        #np.uint([82, 75, 83, 89, 87, 77]):
        ambig, res1, res2 = GETCONS[aidx]

        ## get true wherever tmpseq is ambig
        idx, idy = np.where(tmpseq == ambig)
        halfmask = np.random.choice(np.array([True, False]), idx.shape[0])

        for col in range(idx.shape[0]):
            if halfmask[col]:
                tmpseq[idx[col], idy[col]] = res1
            else:
                tmpseq[idx[col], idy[col]] = res2
    return tmpseq
