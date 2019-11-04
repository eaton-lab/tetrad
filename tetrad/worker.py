#!/usr/bin/env python

""" 
Wrap methods around jitted funcs to run single threaded. 

Uses code from this great source:
https://www.kaggle.com/sggpls/singlethreaded-instantgratification
"""

import sys
import h5py
import ctypes
import numpy as np

from .utils import TESTS
from .jitted import calculate
from .threading import single_threaded


# currently deprecated !!!!!!!!!!!!!
def worker(tet, chunk):

    # open seqarray view, the modified arr is in bootstarr
    with h5py.File(tet.files.idb, 'r') as io5:
        seqview = io5["bootsarr"][:]
        maparr = io5["bootsmap"][:, :0]
        smps = io5["quartets"][chunk:chunk + tet._chunksize]

        # create an N-mask array of all seq cols
        nall_mask = seqview[:] == 78

    # init arrays to fill with results
    rquartets = np.zeros((smps.shape[0], 4), dtype=np.uint16)
    rinvariants = np.zeros((smps.shape[0], 16, 16), dtype=np.uint16)

    # fill arrays with results as we compute them. 
    for idx in range(smps.shape[0]):
        sidx = smps[idx]
        seqs = seqview[sidx]
        nmask = nall_mask[sidx]

        # mask invariant or contains missing
        mask0 = np.any(nmask, axis=0)
        mask1 = np.all(seqs == seqs[0], axis=0)
        mapmask = np.invert(mask0 | mask1)

        # here are the jitted funcs
        bidx, invar = calculate(seqs, maparr, mapmask, TESTS)

        # store results
        rquartets[idx] = smps[idx][bidx]
        rinvariants[idx] = invar        


def nworker(tet, chunk):
    """
    Worker to distribute work to jit funcs. Wraps everything on an 
    engine to run single-threaded to maximize efficiency for 
    multi-processing.
    """
    # set the thread limit on the remote engine to 1 for multiprocessing
    # oldlimit = set_thread_limit(1)

    # single thread limit
    with single_threaded(np):

        # open seqarray view, the modified arr is in bootstarr
        with h5py.File(tet.files.idb, 'r') as io5:
            seqview = io5["bootsarr"][:]
            maparr = io5["bootsmap"][:, 0]
            smps = io5["quartets"][chunk:chunk + tet._chunksize]

            # create an N-mask array of all seq cols
            nall_mask = seqview[:] == 78

        # init arrays to fill with results
        rquartets = np.zeros((smps.shape[0], 4), dtype=np.uint16)
        rinvariants = np.zeros((smps.shape[0], 16, 16), dtype=np.uint16)
        nsnps = np.zeros(smps.shape[0])

        # TODO: test again numbafying the loop below, but on a super large 
        # matrix. Maybe two strategies should be used for different sized 
        # problems... LOL at this and the note below.

        # fill arrays with results as we compute them. This iterates
        # over all of the quartet sets in this sample chunk. It would
        # be nice to have this all numbified (STOP TRYING...)
        for idx in range(smps.shape[0]):
            sidx = smps[idx]
            seqs = seqview[sidx]

            # these axis calls cannot be numbafied, but I can't 
            # find a faster way that is JIT compiled, and I've
            # really, really, really tried. Tried again now that
            # numba supports axis args for np.sum. Still can't 
            # get speed improvements by numbifying this loop.
            # tried guvectorize too...
            nmask = np.any(nall_mask[sidx], axis=0)
            nmask += np.all(seqs == seqs[0], axis=0) 

            # skip calc and choose a random matrix if no SNPs
            nsnps[idx] = seqs[:, np.invert(nmask)].shape[1]

            # here are the jitted funcs
            if nsnps[idx]:
                bidx, invar = calculate(seqs, maparr, nmask, TESTS)
            else:
                bidx = np.random.randint(3)
                invar = np.zeros((16, 16), dtype=np.uint32)

            # store results
            rquartets[idx] = sidx[bidx]
            rinvariants[idx] = invar

        # old thread limit restored on closed context

    # return results...
    return rquartets, rinvariants, nsnps.mean()
