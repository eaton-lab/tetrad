#!/usr/bin/env python

"""Init the Tetrad class object databases.

"""

from typing import TypeVar
from loguru import logger
# logger = logger.bind(name="tetrad")

Tetrad = TypeVar("Tetrad")




def _get_chunksize(self, ncpus):
    """ 
    Find all quartets of samples and store in a large array
    A chunk size is assigned for sampling from the array of quartets
    based on the number of cpus available. This should be relatively 
    large so that we don't spend a lot of time doing I/O, but small 
    enough that jobs finish often for checkpointing.
    """
    # chunking minimizes RAM on each engine and allows checkpoint progress.
    breaks = 2
    if self.params.nquartets < 5000:
        breaks = 1
    if self.params.nquartets > 100000:
        breaks = 8
    if self.params.nquartets > 500000:
        breaks = 16
    if self.params.nquartets > 5000000:
        breaks = 32

    # incorporate n engines into chunksize
    self._chunksize = max([
        1, 
        sum([
            (self.params.nquartets // (breaks * ncpus)),
            (self.params.nquartets % (breaks * ncpus)),
        ])
    ])


def _init_odb(self):
    # create database for saving resolved quartet sets (ad|bc)
    io5 = h5py.File(self.files.odb, 'w')
    io5.create_dataset(
        name="quartets", 
        shape=(self.params.nquartets, 4), 
        dtype=np.uint16, 
        chunks=(self._chunksize, 4),
    )
    io5.create_group("invariants")
    io5.close()


def _init_idb_quartets(self, force):
    # create database for saving raw quartet sets (a,b,c,d)
    io5 = h5py.File(self.files.idb, 'a')
    if io5.get("quartets"):
        if force:
            del io5["quartets"]
        else:
            raise TetradError(
                "database already exists, use force flag to overwrite.")
    io5.create_dataset(
        name="quartets", 
        shape=(self.params.nquartets, 4), 
        dtype=np.uint16, 
        chunks=(self._chunksize, 4),
        compression='gzip',
    )
    io5.close()

    # submit store job to write into self.database.input
    self._print("initializing quartet sets database")
    if self._fullsampled:
        store_all(self)
    else:
        store_random(self)
        # store_equal(self)


if __name__ == "__main__":

    
    import toytree
    import ipcoal
    TRE = toytree.rtree.unittree(12, treeheight=1e7, seed=123)
    MOD = ipcoal.Model(TRE, Ne=2e6, nsamples=2)
    MOD.sim_snps(1000)
    MOD.write_snps_to_hdf5(name="test", outdir="/tmp", diploid=True)
    # /tmp/test.snps.hdf5

    # tet = Tetrad()
    # init_seqarray()
