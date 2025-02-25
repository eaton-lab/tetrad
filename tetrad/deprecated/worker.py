#!/usr/bin/env python

""" 
Wrap methods around jitted funcs to run single threaded. 

Uses code from this great source:
https://www.kaggle.com/sggpls/singlethreaded-instantgratification
"""

import h5py
import numpy as np
from numba import njit

# from .utils import TESTS
# from .jitted import calculate
# from .threading import single_threaded



@njit
def sub_chunk_to_matrices(narr, mapcol, mask):
    """
    Subsample a single SNP per locus and skip masked sites.
    """

    mats = np.zeros((3, 16, 16), dtype=np.uint32)

    # fill the first mat
    last_loc = np.uint32(-1)
    for idx in range(mapcol.shape[0]):
        if not mask[idx]:
            if not mapcol[idx] == last_loc:
                i = narr[:, idx]
                mats[0, (4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1      
                last_loc = mapcol[idx]

    # fill the alternates
    x = np.uint8(0)
    for y in np.array([0, 4, 8, 12], dtype=np.uint8):
        for z in np.array([0, 4, 8, 12], dtype=np.uint8):
            mats[1, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4)
            mats[2, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4).T
            x += np.uint8(1)

    return mats


@njit
def full_chunk_to_matrices(narr, mapcol, mask):
    """
    Count ALL SNPs but skip masked sites.
    """

    mats = np.zeros((3, 16, 16), dtype=np.uint32)

    # fill the first mat
    for idx in range(mapcol.shape[0]):
        if not mask[idx]:
            i = narr[:, idx]
            mats[0, (4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1      

    # fill the alternates
    x = np.uint8(0)
    for y in np.array([0, 4, 8, 12], dtype=np.uint8):
        for z in np.array([0, 4, 8, 12], dtype=np.uint8):
            mats[1, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4)
            mats[2, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4).T
            x += np.uint8(1)

    return mats



#@njit
def infer_resolved_quartets(idb, start, end, subsample):
    """
    Takes a chunk of quartet sets and returns the inferred quartet along
    with the invariants array and SNPs per quartet.
    """
    # open seqarray view, the modified arr is in bootstarr
    with h5py.File(idb, 'r') as io5:
        seqview = io5["bootsarr"][:]
        maparr = io5["bootsmap"][:]
        smps = io5["quartets"][start:end]

    # select function based on subsample bool
    if subsample:
        count_matrix = sub_chunk_to_matrices
    else:
        count_matrix = full_chunk_to_matrices

    # the three indexed resolutions of each quartet
    TIDXS = np.array([
        [0, 1, 2, 3], 
        [0, 2, 1, 3], 
        [0, 3, 1, 2]], dtype=np.uint8,
    )
    TESTS = np.array([0, 1, 2])

    # mats to fill
    rquartets = np.zeros((smps.shape[0], 4), dtype=np.uint16)
    rinvariants = np.zeros((smps.shape[0], 16, 16), dtype=np.uint16)
    rnsnps = np.zeros(smps.shape[0])

    # iterate over quartet sets
    for idx in range(smps.shape[0]):

        # get quartet
        sidx = smps[idx]

        # get seqs of this quartet
        seqs = seqview[sidx]

        # mask sites with missing
        sums = np.sum(seqs, axis=0)
        nmask = sums > 70

        # mask invariant sites
        nmask += np.sum(seqs == seqs[0], axis=0) == 4    

        # count SNPs into 3x16x16 arrays
        cmats = count_matrix(seqs, maparr[:, 0], nmask)

        # skip if seqs is empty
        nsnps = cmats[0].sum()
        if not nsnps:
            qorder = TIDXS[np.random.randint(3)]        
        else:
            # empty arrs to fill
            svds = np.zeros((3, 16), dtype=np.float64)
            scor = np.zeros(3, dtype=np.float64)
            rank = np.zeros(3, dtype=np.float64)

            # svd and rank.
            for test in TESTS:
                svds[test] = np.linalg.svd(cmats[test].astype(np.float64))[1]
                rank[test] = np.linalg.matrix_rank(cmats[test].astype(np.float64))

            # get minrank, or 11 (TODO: can apply seq model here)
            minrank = int(min(10, rank.min()))
            for test in TESTS:
                scor[test] = np.sqrt(np.sum(svds[test, minrank:]**2))

            # sort to find the best qorder
            qorder = TIDXS[np.argmin(scor)]

        # store results
        rquartets[idx] = sidx[qorder]
        rinvariants[idx] = cmats[0]
        rnsnps[idx] = nsnps

    return rquartets, rinvariants, rnsnps
















# def nworker(tet, chunk):
#     """
#     Worker to distribute work to jit funcs. Wraps everything on an 
#     engine to run single-threaded to maximize efficiency for 
#     multi-processing.
#     """
#     # set the thread limit on the remote engine to 1 for multiprocessing
#     # oldlimit = set_thread_limit(1)

#     # single thread limit
#     with single_threaded(np):

#         # open seqarray view, the modified arr is in bootstarr
#         with h5py.File(tet.files.idb, 'r') as io5:
#             seqview = io5["bootsarr"][:]
#             maparr = io5["bootsmap"][:, 0]
#             smps = io5["quartets"][chunk:chunk + tet._chunksize]

#             # create an N-mask array of all seq cols
#             nall_mask = seqview[:] == 78

#         # init arrays to fill with results
#         rquartets = np.zeros((smps.shape[0], 4), dtype=np.uint16)
#         rinvariants = np.zeros((smps.shape[0], 16, 16), dtype=np.uint16)
#         nsnps = np.zeros(smps.shape[0])

#         # TODO: test again numbafying the loop below, but on a super large 
#         # matrix. Maybe two strategies should be used for different sized 
#         # problems... LOL at this and the note below.

#         # fill arrays with results as we compute them. This iterates
#         # over all of the quartet sets in this sample chunk. It would
#         # be nice to have this all numbified (STOP TRYING...)
#         for idx in range(smps.shape[0]):
#             sidx = smps[idx]
#             seqs = seqview[sidx]

#             # these axis calls cannot be numbafied, but I can't 
#             # find a faster way that is JIT compiled, and I've
#             # really, really, really tried. Tried again now that
#             # numba supports axis args for np.sum. Still can't 
#             # get speed improvements by numbifying this loop.
#             # tried guvectorize too...
#             nmask = np.any(nall_mask[sidx], axis=0)
#             nmask += np.all(seqs == seqs[0], axis=0) 

#             # skip calc and choose a random matrix if no SNPs
#             nsnps[idx] = seqs[:, np.invert(nmask)].shape[1]

#             # here are the jitted funcs
#             if nsnps[idx]:
#                 bidx, invar = calculate(seqs, maparr, nmask, TESTS)
#             else:
#                 bidx = TESTS[np.random.randint(3)]
#                 invar = np.zeros((16, 16), dtype=np.uint32)

#             # store results
#             rquartets[idx] = sidx[bidx]
#             rinvariants[idx] = invar

#         # old thread limit restored on closed context

#     # return results...
#     return rquartets, rinvariants, nsnps.mean()




# # currently deprecated !!!!!!!!!!!!!
# def worker(tet, chunk):

#     # open seqarray view, the modified arr is in bootstarr
#     with h5py.File(tet.files.idb, 'r') as io5:
#         seqview = io5["bootsarr"][:]
#         maparr = io5["bootsmap"][:, :0]
#         smps = io5["quartets"][chunk:chunk + tet._chunksize]

#         # create an N-mask array of all seq cols
#         nall_mask = seqview[:] == 78

#     # init arrays to fill with results
#     rquartets = np.zeros((smps.shape[0], 4), dtype=np.uint16)
#     rinvariants = np.zeros((smps.shape[0], 16, 16), dtype=np.uint16)

#     # fill arrays with results as we compute them. 
#     for idx in range(smps.shape[0]):
#         sidx = smps[idx]
#         seqs = seqview[sidx]
#         nmask = nall_mask[sidx]

#         # mask invariant or contains missing
#         mask0 = np.any(nmask, axis=0)
#         mask1 = np.all(seqs == seqs[0], axis=0)
#         mapmask = np.invert(mask0 | mask1)

#         # here are the jitted funcs
#         bidx, invar = calculate(seqs, maparr, mapmask, TESTS)

#         # store results
#         rquartets[idx] = smps[idx][bidx]
#         rinvariants[idx] = invar        
