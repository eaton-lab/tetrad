#!/usr/bin/env python

"""Write the database HDF5 file.

This is called by tetrad init to create the HDF5 database.

TODO
----
- allow imap sampling here ...
-
"""

from pathlib import Path
from itertools import chain
import h5py
from loguru import logger
import numpy as np
from pandas import read_csv
from scipy.special import comb
from tetrad.jit.get_spans import jit_get_spans
from tetrad.jit.resolve_ambigs import jit_resolve_ambigs

# assert ".snps.hdf5" in str(data), f"`data` ({data}) is not .snps.hdf5"
# logger.info(f"loading snps array: {len(samples)} samples x {nsnps} SNPs")


def deprecated_get_names_from_database(data: Path, imap: dict[str, list[str]]) -> dict[int, str]:
    """Return dict of {index: name} in order from HDF5, subset by imap.

    The imap will subselect names to be includd in the analysis, but
    does not relabel the names. Keeps all names in the values.
    Example IMAP: {'a': ['a_1', 'a_2'], 'b': ['b_1', 'b_2'], ...}

    Note: it will be faster to create a HDF5 that only includes your
    selected samples, rather than subsample from it, as most of the
    computation time is spent finding shared data in the matrix. So
    for now we do not implement an IMAP method.
    """
    with h5py.File(data, 'r') as io5:
        # get names as list[str]
        names = [
            i.decode("utf-8") if isinstance(i, bytes) else i
            for i in io5["snps"].attrs["names"]
        ]
        # create dict mapping data index to names
        samples = dict(enumerate(names))

        # subsample names from imap
        if imap:
            if isinstance(imap, Path):
                imap = imap_tsv_to_dict(imap)
            keep = list(chain(*[j for (i, j) in imap.items()]))
            samples = {i: j for (i, j) in samples.items() if j in keep}
    return samples


def get_names_from_database(data: Path) -> dict[int, str]:
    """Return dict of {index: name} in order from HDF5, subset by imap.

    The imap will subselect names to be includd in the analysis, but
    does not relabel the names. Keeps all names in the values.
    Example IMAP: {'a': ['a_1', 'a_2'], 'b': ['b_1', 'b_2'], ...}
    """
    with h5py.File(data, 'r') as io5:
        # get names as list[str]
        names = [
            i.decode("utf-8") if isinstance(i, bytes) else i
            for i in io5["snps"].attrs["names"]
        ]
        # create dict mapping data index to names
        samples = dict(enumerate(names))
    return samples


def get_nsnps_from_database(data: Path) -> int:
    """Return the number of SNPs in the HDF5 database."""
    with h5py.File(data, 'r') as io5:
        return io5['snps'].shape[1]


def get_nquartets(nsamples: int, nquartets: int) -> tuple[int, int]:
    """Return n quartets to sample and the total number of quartets"""

    # this is the minimum we should sample
    rough = int(nsamples ** 2.8)

    # this is the total number we could sample
    total = int(comb(nsamples, 4))

    # raise exception if nsamples is too large for our precision
    assert total < 4_294_967_295, "max possible quartets exceeded."

    # default is to sample all quartets?
    if not nquartets:
        logger.info(f"quartet sampler [full]: {total}/{total}")
        return total, total

    # warn user if they entered a low value for nquartets
    if nquartets < rough:
        logger.warning(f"nquartets is low ({nquartets}/{total}), consider raising to {rough} or higher")
        return nquartets, total
    elif nquartets > total:
        logger.info(f"quartet sampler [full]: {total}/{total}")
        return total, total
    else:
        logger.info(f"quartet sampler [random]: {nquartets}/{total}")
        return nquartets, total


def init_database(data: Path, out: Path, nsnps: int, nsamples: int, nquartets: int, rng: int) -> None:
    """Samples sequence alignment from 'data' -> 'idb'.

    /seqarr: A copy of orig data, for restarting w/o needing it.
    /spans: spans of loci for subsampling SNPs.
    /tmpmap: snps to locus mapping of resampled loci.
    /tmparr: snps on resampled loci.
    /quartets: results.

    Subset samples from IMAP. If no imap then imap matches samples
    as dict. Fills the seqarr with the full SNP data set while
    keeping memory requirements super low, and creates a bootsarr
    copy with the following modifications:

    1) converts "-" into "N"s, since they are similarly treated as missing.
    2) randomly resolve ambiguities (RSKWYM)
    3) convert to uint8 for smaller memory load and faster computation.
    """
    # ...
    rng = np.random.default_rng(rng)

    # ...
    with h5py.File(data, 'r') as io5:
        with h5py.File(out, 'w', libver="latest") as idb:

            # load the snpsmap which records SNPs on the same locus.
            # [locidx-1-indexed, snpidx-0-indexed, position-1-indexed, None, total-SNPs-1-indexed]
            # convert snpsmap to [0-indexed locidx, total-0-indexed]
            snpsmap = io5["snpsmap"][:]
            snpsmap[:, 0] = snpsmap[:, 0] - 1
            snpsmap[:, 1] = np.arange(nsnps)
            snpsmap = snpsmap[:, :2]

            # spans stores all the snpmap info needed for original data
            idb.create_dataset("spans", data=jit_get_spans(snpsmap), dtype=np.int64)

            # tmpmap stores a mapping of loci to SNPs. It starts with
            # original mapping but is updated each replicate with the
            # mapping of new re-sampled datasets.
            idb.create_dataset("tmpmap", data=snpsmap, dtype=np.uint32)
            nloci = np.unique(snpsmap[:, 0]).size
            logger.info(f"max unlinked SNPs per quartet [nloci]: {nloci}")

            # load the SNPs matrix, set - to N (78), and write to
            # 'seqarr' dataset: shape=(len(self.samples), self.nsnps).
            # we DO NOT resolve ambiguities in this array, since that
            # is done later randomly after SNPs are subsampled.
            tmpseq = io5["snps"][:].astype(np.uint8)
            tmpseq[tmpseq == 45] = 78
            idb.create_dataset("seqarr", data=tmpseq, dtype=np.uint8)

            # boot samples: resolve ambigs and converts CATG bases to
            # matrix indices: 65,67,71,84 -> 0,1,2,3
            tmpseq = jit_resolve_ambigs(tmpseq, seed=rng.integers(2**31))
            tmpseq[tmpseq == 65] = 0
            tmpseq[tmpseq == 67] = 1
            tmpseq[tmpseq == 71] = 2
            tmpseq[tmpseq == 84] = 3
            idb.create_dataset("tmparr", data=tmpseq, dtype=np.uint8)

            ## dataset for saving resolved quartet sets (ad|bc)
            # could be (nquart, 4, nboots+1) to store ALL results...
            # idb.create_dataset("quartets", shape=(nquartets, 4), dtype=np.uint16)
        del tmpseq
        del snpsmap
    logger.info(f"wrote database file to {out}")
    return out


def write_database(project: "Project") -> "Project":
    """..."""
    # get {int: name} map of samples optionally subselected by imap
    samples = get_names_from_database(project.data)
    nsamples = len(samples)
    nsnps = get_nsnps_from_database(project.data)
    nqrts, nqrts_total = get_nquartets(nsamples, project.nquartets)
    # logger.info(nsamples)
    # logger.info(nsnps)
    # logger.info(nqrts)
    # logger.info(project.random_seed)
    init_database(project.data, project.database_file, nsnps, nsamples, nqrts, project.random_seed)
    project.nqrts = nqrts
    project.nqrts_total = nqrts_total
    project.nsamples = nsamples
    project.nsnps = nsnps
    project.samples = samples


def imap_tsv_to_dict(imap: Path) -> dict[str, list[str]]:
    """return IMAP dict parsed from TSV file of clade\tsample\n"""
    imap = read_csv(imap, sep=r"\s+", header=None)
    return imap.groupby(0)[1].apply(list).to_dict()


if __name__ == "__main__":


    # DATA = Path("/home/deren/Documents/tools/tetrad/lib123-Ahypo-ref.snps.hdf5")
    DATA = Path("/home/deren/Documents/tools/tetrad/new_all_delphinium_123_3x.snps.hdf5")
    IMAP = Path("") #"/home/deren/Documents/tools/tetrad/sample-to-species-map2.tsv")
    # ../../imap.tsv")
    # print(imap_tsv_to_dict(IMAP))

    # DATA = Path("/tmp/small.snps.hdf5")
    OUT = Path("/tmp/small.database.hdf5")

    if not DATA.exists():
        raise IOError(f"data does not exist: {DATA}")
    else:
        logger.info(f"{DATA} exists: {DATA.exists()}")
    if not IMAP.exists():
        IMAP = None
    else:
        logger.info(f"{DATA} exists: {DATA.exists()}")

    # get dict mapping {sidx: name, ...}
    samples = get_names_from_database(DATA)
    nsamples = len(samples)
    nsnps = get_nsnps_from_database(DATA)
    nqrts, nqtotal = get_nquartets(nsamples, 0)

    init_database(DATA, OUT, nsnps, nsamples, nqrts, 123)
    # raise SystemExit(0)
    with h5py.File(OUT, 'r') as io5:
        for name in ["tmpmap", "tmparr", "spans", "seqarr"]:
            db = io5[name]
            print(f"{name} {db.shape} {db.dtype}\n{db[:]}\n-------------")
