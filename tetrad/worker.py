#!/usr/bin/env python

""" 
Summarize results
"""

import sys
import h5py
import ctypes
import numpy as np

from .utils import TESTS
from .jitted import calculate


def set_thread_limit(cores):
    """
    set mkl thread limit and return old value so we can reset
    when finished. 
    """
    try:
        if "linux" in sys.platform:
            mkl_rt = ctypes.CDLL('libmkl_rt.so')
        else:
            mkl_rt = ctypes.CDLL('libmkl_rt.dylib')
    
        # get old limit, set new limit, and return old
        oldlimit = mkl_rt.mkl_get_max_threads()
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
        return oldlimit

    except OSError:
        if "linux" in sys.platform:
            openblas_rt = ctypes.CDLL('openblas_rt.so')
        else:
            openblas_rt = ctypes.CDLL('openblas_rt.dylib')

        # get old limit, set new limit, and return old
        oldlimit = openblas_rt.openblas_get_max_threads()
        openblas_rt.openblas_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
        return oldlimit



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
    oldlimit = set_thread_limit(1)

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

    # fill arrays with results as we compute them. This iterates
    # over all of the quartet sets in this sample chunk. It would
    # be nice to have this all numbified (STOP TRYING...)
    for idx in range(smps.shape[0]):
        sidx = smps[idx]
        seqs = seqview[sidx]

        ## these axis calls cannot be numbafied, but I can't 
        ## find a faster way that is JIT compiled, and I've
        ## really, really, really tried. Tried again now that
        ## numba supports axis args for np.sum. Still can't 
        ## get speed improvements by numbifying this loop.
        ## tried guvectorize too...
        nmask = np.any(nall_mask[sidx], axis=0)
        nmask += np.all(seqs == seqs[0], axis=0) 

        ## here are the jitted funcs
        bidx, invar = calculate(seqs, maparr, nmask, TESTS)

        ## store results
        rquartets[idx] = smps[idx][bidx]
        rinvariants[idx] = invar

    # reset thread limit
    set_thread_limit(oldlimit)

    # return results...
    return rquartets, rinvariants
