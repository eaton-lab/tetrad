#!/usr/bin/env python

"""
Wraps jitted methods so numpy will run single-threaded for multi-processing.
"""

import subprocess as sps
import re
import sys
import os
import glob
import warnings
import ctypes


_MKL_ = 'mkl'
_OPENBLAS_ = 'openblas'


class BLAS:
    """
    A class for getting and setting threadlimits for OPENBLAS or MKL
    """
    def __init__(self, cdll, kind):

        # check string type
        if kind not in (_MKL_, _OPENBLAS_):
            raise ValueError(
                "kind must be {} or {}, got {} instead"
                .format(_MKL_, _OPENBLAS_, kind)
                )

        # store attrs       
        self.kind = kind
        self.cdll = cdll
        
        # clone cdll set and get funcs
        if kind == _MKL_:
            self.get_n_threads = cdll.MKL_Get_Max_Threads
            self.set_n_threads = cdll.MKL_Set_Num_Threads
        else:
            self.get_n_threads = cdll.openblas_get_num_threads
            self.set_n_threads = cdll.openblas_set_num_threads
            


def get_blas(numpy_module):
    """
    Finds the BLAS CDL of MKL (or OPENBLAS)
    """

    # simple expressio matching to get ...
    LDD = 'ldd'
    LDD_PATTERN = r'^\t(?P<lib>.*{}.*) => (?P<path>.*) \(0x.*$'

    # get lib from numpy path and call ldd to see shared library dependencies
    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, '_multiarray_umath.*so'))[0]
    proc = sps.Popen([LDD, MULTIARRAY_PATH], stdout=sps.PIPE)
    ldd_result = proc.communicate()[0].decode()

    # collect results from ldd call
    if _MKL_ in ldd_result:
        kind = _MKL_
    elif _OPENBLAS_ in ldd_result:
        kind = _OPENBLAS_
    else:
        return

    # get the CDLL and return BLAS class object
    pattern = LDD_PATTERN.format(kind)
    match = re.search(pattern, ldd_result, flags=re.MULTILINE)
    if match:
        lib = ctypes.CDLL(match.groupdict()['path'])
        return BLAS(lib, kind)
    


class single_threaded:
    """
    A with wrapper to run single-threaded numpy code.
    """
    def __init__(self, numpy_module):

        # check the numpy module
        self.blas = get_blas(numpy_module)
        self.old_n_threads = None


    def __enter__(self):      
        # set thread-count to 1 unless library is missing then freak out.
        if self.blas is None:
            pass
            # TODO: LOG FILE 
            #warnings.warn(
            #    'No MKL/OpenBLAS found, assuming NumPy is single-threaded.'
            #)
        else:
            self.old_n_threads = self.blas.get_n_threads()
            self.blas.set_n_threads(1)


    def __exit__(self, *args):
        # reset blas to its starting settings
        if self.blas is not None:
            self.blas.set_n_threads(self.old_n_threads)
            if self.blas.get_n_threads() != self.old_n_threads:
                raise RuntimeError(
                    'Failed to reset {} to {} threads (previous value).'
                    .format(self.blas.kind, self.old_n_threads)
                )
    
    def __call__(self, func):
        def _func(*args, **kwargs):
            self.__enter__()
            func_result = func(*args, **kwargs)
            self.__exit__()
            return func_result
        return _func