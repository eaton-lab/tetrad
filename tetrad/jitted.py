#!/usr/bin/env python

"""
Just in time compiled functions for tetrad
"""
from builtins import range
import numpy as np
from numba import njit
from .utils import GETCONS


@njit(parallel=True)
def subsample_loci_and_snps(snpsmap, seed, resample_loci):
    "Subsample loci and SNPs (one per locus) using snpsmap"
    
    # initialize numba random seed 
    np.random.seed(seed)

    # get locus idxs
    sidxs = np.unique(snpsmap[:, 0])

    # resample locus idxs
    if resample_loci:
        sidxs = np.random.choice(sidxs, sidxs.size)

    # an array to fill
    subs = np.zeros(sidxs.size, dtype=np.int64)
    idx = 0

    # iterate over loci 
    for sidx in sidxs:

        # get all sites in this locus
        sites = snpsmap[snpsmap[:, 0] == sidx, 1]

        # randomly select one and save
        site = np.random.choice(sites)
        subs[idx] = site
        idx += 1

    # return as sorted array of int64
    subs.sort()
    return subs


@njit()
def calculate(seqnon, mapcol, nmask, tests):
    """
    Groups together several other numba funcs.
    """
    # get the invariants matrix
    mats = chunk_to_matrices(seqnon, mapcol, nmask)

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


@njit
def old_jget_spans(maparr, spans):
    """ 
    Get span distance for each locus in original seqarray. This
    is used to create re-sampled arrays in each bootstrap to sample
    unlinked SNPs. 
    """
    ## start at 0, finds change at 1-index of map file
    bidx = 1
    spans = np.zeros((maparr[-1, 0], 2), np.uint64)
    ## read through marr and record when locus id changes
    for idx in range(1, maparr.shape[0]):
        cur = maparr[idx, 0]
        if cur != bidx:
            # idy = idx + 1
            spans[cur - 2, 1] = idx
            spans[cur - 1, 0] = idx
            bidx = cur
    spans[-1, 1] = maparr[-1, -1]
    return spans


@njit()
def jget_spans(maparr):
    """ 
    Returns array with span distances for each locus in original seqarray. 
    [ 0, 33],
    [33, 47],
    [47, 51], ...
    """
    sidx = 0
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
            if lidx:
                spans[lidx] = spans[lidx - 1, 1], idx
            else:
                spans[lidx] = np.array((0, idx))
            lidx += 1
            sidx = locs[lidx]

    # final end span
    spans[-1] = np.array((spans[-2, 1], maparr[-1, -1] + 1))
    return spans


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
