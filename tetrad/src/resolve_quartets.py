#!/usr/bin/env python

"""Process SNP data for quartets.

Extracts sequence data given indices (0, 1, 2, 3) and returns the 
resolved quartet relationship, ordered as 03|14 -> (0, 3, 1, 4).
"""

from pathlib import Path
import h5py
import numpy as np
from numba import njit

TESTS = np.array([0, 1, 2])


def infer_resolved_quartets(
    database: Path,
    nsamples: int,
    qrts: list[tuple[int, int, int, int]],
    subsample_snps: bool = True,
    ) -> np.ndarray:
    """Return a chunk of resolved quartets in a matrix.

    Loads the quartets and data on the remote process and computes.
    """
    # expand the quartet generator and convert to an array
    qrts = np.array(list(qrts), dtype=np.uint32)

    # load SNPs from database (using single-writer multiple reader mode)
    # the 'tmp' arrays contain the ambig-resolved SNPs from the full or
    # bootstrap sampled data, depending on the iteration.
    with h5py.File(database, 'r', swmr=True) as io5:
        tmparr = io5["tmparr"][:]
        tmpmap = io5["tmpmap"][:]

    # return arrays with resolved qrts, nsnps, and weights
    return new_infer_resolved_quartets(tmparr, tmpmap, qrts, subsample_snps)    
    # return jit_infer_resolved_quartets(tmparr, tmpmap, qrts, subsample_snps)


@njit
def subsample_chunk_to_matrices(
    tmparr: np.ndarray,
    tmpmap: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Count ALL unmasked site patterns and return as site count matrix.
    
    This func DOES filter using tmpmap based on linkage, i.e., 
    SNPs on the same locus will be counted rather than thinned. See
    sub_chunk_to_matrices for sampling one SNP per locus.
    """
    # highest occurrence of a site pattern allowed is 4_294_967_295
    mats = np.zeros((3, 16, 16), dtype=np.uint32)

    # fill the first mat
    last_loc = np.uint32(-1)
    for idx in range(tmpmap.shape[0]):
        if not mask[idx]:
            if not tmpmap[idx] == last_loc:
                i = tmparr[:, idx]
                mats[0, (4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1      
                last_loc = tmpmap[idx]

    # fill the alternates
    x = np.uint8(0)
    for y in np.array([0, 4, 8, 12], dtype=np.uint8):
        for z in np.array([0, 4, 8, 12], dtype=np.uint8):
            mats[1, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4)
            mats[2, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4).T
            x += np.uint8(1)
    return mats


@njit
def full_chunk_to_matrices(
    tmparr: np.ndarray,
    tmpmap: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Count ALL unmasked site patterns and return as site count matrix.
    
    This func does not filter using tmpmap based on linkage, i.e., 
    SNPs on the same locus will be counted rather than thinned. See
    sub_chunk_to_matrices for sampling one SNP per locus.
    """
    # highest occurrence of a site pattern allowed is 4_294_967_295
    mats = np.zeros((3, 16, 16), dtype=np.uint32)

    # fill the first mat
    for idx in range(tmpmap.shape[0]):
        if not mask[idx]:
            i = tmparr[:, idx]
            mats[0, (4 * i[0]) + i[1], (4 * i[2]) + i[3]] += 1      

    # fill the alternates
    x = np.uint8(0)
    for y in np.array([0, 4, 8, 12], dtype=np.uint8):
        for z in np.array([0, 4, 8, 12], dtype=np.uint8):
            mats[1, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4)
            mats[2, y:y + np.uint8(4), z:z + np.uint8(4)] = mats[0, x].reshape(4, 4).T
            x += np.uint8(1)
    return mats


# def jit_infer_resolved_quartets(
#     tmparr: np.ndarray,
#     tmpmap: np.ndarray,
#     quartets: np.ndarray,
#     subsample_snps: bool,
# ):
#     """calls jit-compiled code for infer_resolved_quartets.

#     Returns three arrays: resolved quartets, number of SNPs, and the
#     weight (confidence) in the resolution. Note that weights to wQMC
#     can be any real number. Here we scale weights 0-1, but it would
#     be useful to test alternative weighting schemes.
#     """
#     # result matrices to fill
#     rnsnps = np.zeros(quartets.shape[0])
#     results = quartets.copy()
#     results[:] = 0

#     # track and return scores while debugging
#     scores = np.zeros(quartets.shape[0])

#     for qidx in range(quartets.shape[0]):
#         # get sample indices for this quartet (0, 1, 2, 3)
#         sidx = quartets[qidx]
#         # get sequence array rows for just these four samples
#         seqs = tmparr[sidx, :]
#         # mask sites with missing values (78) among these four samples
#         # numba doesn't support 'any' over axes, so we use sum which is
#         # a bit slower, but faster in the context of keeping all jitted.
#         nmask0 = np.sum(seqs >= 78, axis=0)
#         # mask sites that are invariant among these four samples
#         nmask1 = np.sum(seqs == seqs[0], axis=0) == 4
#         # count SNP patterns and shape into 3x16x16 matrix.
#         if subsample_snps:
#             cmats = subsample_chunk_to_matrices(seqs, tmpmap[:, 0], nmask0 + nmask1)
#         else:
#             cmats = full_chunk_to_matrices(seqs, tmpmap[:, 0], nmask0 + nmask1)

#         # find the best resolution of subtree based on matrices
#         nsnps = cmats[0].sum()

#         # no data, ...
#         # FIXME: is random return better than exclusions?? needs testing.
#         if not nsnps:
#             qorder = TIDXS[np.random.randint(3)]
#             scores[qidx] = 0.001 # 1 / 3

#         # data is present, find solution.
#         else:
#             # empty arrs to fill
#             svds = np.zeros((3, 16), dtype=np.float64)
#             scor = np.zeros(3, dtype=np.float64)
#             rank = np.zeros(3, dtype=np.float64)

#             # svd and rank.
#             for test in TESTS:
#                 svds[test] = np.linalg.svd(cmats[test].astype(np.float64))[1]
#                 rank[test] = np.linalg.matrix_rank(cmats[test].astype(np.float64))

#             # get minrank, or 11 (TODO: rank can apply different seq models here)
#             minrank = int(min(10, rank.min()))
#             for test in TESTS:
#                 scor[test] = np.sqrt(np.sum(svds[test, minrank:]**2))

#             # sort to find the best qorder
#             bidx = np.argmin(scor)
#             qorder = TIDXS[bidx]

#             # score can be weighted in two ways: 
#             # 1. How much better is best matrix (relative weight of matrix 1)
#             # 2. How symmetric are the other two (D-statistic)
#             sum_score = scor.sum()
#             if not sum_score:
#                 scores[qidx] = 0.001 # 1 / 3.
#             else:
#                 scores[qidx] = 1 - (scor[bidx] / scor.sum())

#         # store results
#         results[qidx] = sidx[qorder]
#         rnsnps[qidx] = nsnps
#     return results, rnsnps, scores



def new_infer_resolved_quartets(
    tmparr: np.ndarray,
    tmpmap: np.ndarray,
    quartets: np.ndarray,
    subsample_snps: bool,
):
    """calls jit-compiled code for infer_resolved_quartets.

    Returns three arrays: resolved quartets, number of SNPs, and the
    weight (confidence) in the resolution. Note that weights to wQMC
    can be any real number. Here we scale weights 0-1, but it would
    be useful to test alternative weighting schemes.
    """
    # result matrices to fill
    rscor = np.zeros((quartets.shape[0], 3), dtype=np.float64)
    rstat = np.zeros((quartets.shape[0], 2), dtype=np.uint32)

    for qidx in range(quartets.shape[0]):
        # get sample indices for this quartet (0, 1, 2, 3)
        sidx = quartets[qidx]
        # get sequence array rows for just these four samples
        seqs = tmparr[sidx, :]
        # mask sites with missing values (78) among these four samples
        # numba doesn't support 'any' over axes, so we use sum which is
        # a bit slower, but faster in the context of keeping all jitted.
        nmask0 = np.sum(seqs >= 78, axis=0)
        # mask sites that are invariant among these four samples
        nmask1 = np.sum(seqs == seqs[0], axis=0) == 4
        # count SNP patterns and shape into 3x16x16 matrix.
        if subsample_snps:
            cmats = subsample_chunk_to_matrices(seqs, tmpmap[:, 0], nmask0 + nmask1)
        else:
            cmats = full_chunk_to_matrices(seqs, tmpmap[:, 0], nmask0 + nmask1)

        # find the best resolution of subtree based on matrices
        nsnps = cmats[0].sum()

        # no data, ...
        # FIXME: is random return better than exclusions?? needs testing.
        if not nsnps:
            rstat[qidx, 0] = np.random.randint(3)
            rscor[qidx] = 0.001 # 1 / 3

        # data is present, find solution.
        else:
            # empty arrs to fill
            svds = np.zeros((3, 16), dtype=np.float64)
            rank = np.zeros(3, dtype=np.float64)

            # svd and rank.
            for test in TESTS:
                svds[test] = np.linalg.svd(cmats[test].astype(np.float64))[1]
                rank[test] = np.linalg.matrix_rank(cmats[test].astype(np.float64))

            # get minrank, or 11 (TODO: rank can apply different seq models here)
            minrank = int(min(10, rank.min()))
            for test in TESTS:
                rscor[qidx, test] = np.sqrt(np.sum(svds[test, minrank:]**2))

            # sort to find the best qorder
            rstat[qidx, 0] = np.argmin(rscor[qidx])

            # score can be weighted in two ways: 
            # 1. How much better is best matrix (relative weight of matrix 1)
            # 2. How symmetric are the other two (D-statistic)
            # sum_score = scor.sum()
            # if not sum_score:
            #     rscor[qidx] = 0.001 # 1 / 3.
            # else:
            #     rscor[qidx] = 1 - (scor[bidx] / scor.sum())

        # store results
        # rqrts[qidx] = sidx[qorder]
        rstat[qidx, 1] = nsnps
    return quartets, rstat, rscor



if __name__ == "__main__":

    import tetrad
    import toytree
    TRE = toytree.rtree.unittree(20, treeheight=5e6, seed=123)
    toytree.utils.show(TRE.draw(layout='unroot')[0])
    DATA = Path("/tmp/test.snps.hdf5")
    TET = tetrad.Tetrad(name='test', workdir="/tmp", data=str(DATA), nquartets=0)
    
    cidx, qrts = next(TET.iter_quartet_chunks())
    rqrts, rnsnps = infer_resolved_quartets(TET.files.database, qrts)
    for i, j in zip(qrts, rqrts):
        print(i, j)
