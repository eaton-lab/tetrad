#!/usr/bin/env python

"""Quartet supertree inference tool.

In each bootstrap replicate one SNP is randomly sampled from each locus
for every quartet set. This is important, since it maximizes sampling SNPs
that are actually segregatng in that quartet. This means subsampling SNPs
does not take place until the worker function step.

Notes
-----
/data
    /snps [names]
    /snpsmap [columns]
    /genos
    /ref..

/database
    /seqarr (uint8 cleaned original snp data)
    /tmparr (uint8 sampled snps for a given rep)
    /tmpmap (uint64 snpsmap loc/site indices of the resampled loci.)
    /spans  (uint64 ordered N snps in loci)
    /quartets (uint16 resolved quartet indices)
    /invariants (uint...

References
----------
- Chifman, J. and L. Kubatko. 2014. Quartet inference from SNP data under
the coalescent, Bioinformatics, 30(23): 3317-3324.

- Chifman, J. and L. Kubatko. 2015. Identifiability of the unrooted species
tree topology under the coalescent model with time-reversible substitution
processes, site-specific rate variation, and invariable sites, Journal of
Theoretical Biology 374: 35-47

"""

from typing import Optional, List, Iterator, Tuple
import time
import itertools
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict

import h5py
import toytree
import numpy as np
from scipy.special import comb
from loguru import logger

from tetrad.utils import TetradError, get_qmc_binary
from tetrad.cluster import Cluster
from tetrad.worker import infer_resolved_quartets
from tetrad.jitted import (
    jit_resolve_ambigs,
    jit_get_spans,
    jit_resample,
)


logger = logger.bind(name="tetrad")


@dataclass
class Files:
    data: Path = ""
    database: Path = ""
    quartets: Path = ""
    tmp: Path = ""

@dataclass
class Trees:
    tree: Path = ""
    cons: Path = ""
    boots: Path = ""
    nhx: Path = ""

@dataclass
class Stats:
    todo: int = 0
    other: float = 0


@dataclass
class Tetrad:
    """Tetrad class object for ...

    A Tetrad class object should be initialized using `start_tetrad`
    or loaded from an existing checkpoint using `load_tetrad`.

    Parameters
    ----------
    data: str
        It is assumed that all samples in the database will be used...
    """
    name: str
    """A str name to use a prefix for output files."""
    data: str
    """Path to a '.vcf', or '.snps.hdf5' formatted SNPs data file."""
    workdir: Path = Path("analysis-tetrad")
    """Directory path where intermediate and result files will be stored."""
    nquartets: int = 0
    """Number of quartets to (sub)sample. 0=all quartets are sampled."""
    nboots: int = 0
    """Number of bootstrap replicates to run."""
    seed: Optional[int] = None
    """If True checkpointed results are loaded from the {workdir}/{name} path."""
    subsample_snps: bool = True
    """If True subsample one SNP per locus (linkage-block), else use all SNPs."""
    # save_invariants: bool = False
    # """..."""
    # resolve_ambigs: bool = True
    # """If True ambiguous IUPAC sites are randomly resolved; False they are excluded."""

    # attrs to be filled
    files: Files = Files()
    """Paths to files including HDF5 database used as tmp storage."""
    trees: Trees = Trees()
    """Paths to output tree files."""
    stats: Stats = Stats()
    """Statistics recorded about the analysis. See also NHX tree."""
    samples: List[str] = field(default_factory=list)
    """List of sample names in the order found in `data`."""

    # private attributes
    _rng: np.random.Generator = np.random.default_rng(seed)
    """rng created at init from new or stored seed."""
    _cli: bool = False
    """If in the CLI then x runs differently?"""
    _chunksize: int = 1
    """Number of quartets to assign to each processor."""
    _qmc: Path = get_qmc_binary()
    """Path to the QMC binary"""

    def __post_init__(self):
        self.workdir = Path(self.workdir).expanduser().absolute()
        self.workdir.mkdir(exist_ok=True)
        self.files.data = Path(self.data).expanduser().absolute()
        self.files.database = self.workdir / Path(self.name + ".database.hdf5")
        self.files.quartets = self.workdir / Path(self.name + ".quartets.txt")
        self.files.tmp = self.workdir / Path(self.name + ".tmp.txt")
        self.trees.tree = self.workdir / Path(self.name + ".tree")
        self.trees.cons = self.workdir / Path(self.name + ".cons")
        self.trees.boots = self.workdir / Path(self.name + ".boots")
        self.trees.nhx = self.workdir / Path(self.name + ".nhx")

        # fills: .samples, .sidxs, ...
        self._clear_old_files()
        self._get_names_from_database()
        self._count_quartets()
        self._set_chunksize()
        self._init_database()

    def _clear_old_files(self):
        """Remove any existing files if starting anew, or some if continuing

        TODO: continuing ...
        """
        for fname, fpath in asdict(self.files).items():
            if fname != "data":
                fpath.unlink(missing_ok=True)
        for fname, fpath in asdict(self.trees).items():
            fpath.unlink(missing_ok=True)

    def _get_names_from_database(self):
        """Extract ordered sample names, and nsnps, from the HDF5 data."""
        # get data shape from io5 input file
        assert ".snps.hdf5" in str(self.files.data), f"`data` ({self.files.data}) is not .snps.hdf5"

        # get sequence data and names from the snps.hdf5 database
        with h5py.File(self.files.data, 'r') as io5:

            # get names from HDF5 (may be strs or bytes, b/c hdf5 versions)
            self.samples = dict(enumerate(
                i.decode("utf-8") if isinstance(i, bytes) else i
                for i in io5["snps"].attrs["names"]))

            self.nsnps = io5['snps'].shape[1]
            logger.info(
                "loading snps array: "
                f"{len(self.samples)} samples x {self.nsnps} SNPs")

    def _count_quartets(self):
        """Sets `self.nquartets` to full or subsampled number."""
        rough = int(len(self.samples) ** 2.8)
        total = int(comb(len(self.samples), 4))
        self.nquartets_total = total
        assert total < 4_294_967_295, "max possible quartets exceeded."

        if not self.nquartets:
            self.nquartets = total
            logger.info(f"quartet sampler [full]: {self.nquartets}/{total}")
        else:
            # warn user if they entered a low value for nquartets
            if self.nquartets < rough:
                logger.warning(
                    f"nquartets is very low ({self.nquartets}/{total}), "
                    "consider setting a higher value.")
            if self.nquartets > total:
                self.nquartets = total
                logger.info(f"quartet sampler [full]: {self.nquartets}/{total}")
            else:
                self.nquartets = int(self.nquartets)
                logger.info(f"quartet sampler [random]: {self.nquartets}/{total}")

    def _init_database(self) -> None:
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
        # data base file to write the transformed array to.
        io5 = h5py.File(self.files.data, 'r')
        with h5py.File(self.files.database, 'w', libver="latest") as idb:

            # load the snpsmap which records SNPs on the same locus.
            # [locidx-1-indexed, snpidx-0-indexed, position-1-indexed, None, total-SNPs-1-indexed]
            # convert snpsmap to [0-indexed locidx, total-0-indexed]
            snpsmap = io5["snpsmap"][:]
            snpsmap[:, 0] = snpsmap[:, 0] - 1
            snpsmap[:, 1] = np.arange(self.nsnps)
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
            tmpseq = jit_resolve_ambigs(tmpseq, seed=self._rng.integers(2**31))
            tmpseq[tmpseq == 65] = 0
            tmpseq[tmpseq == 67] = 1
            tmpseq[tmpseq == 71] = 2
            tmpseq[tmpseq == 84] = 3
            idb.create_dataset("tmparr", data=tmpseq, dtype=np.uint8)

            ## dataset for saving resolved quartet sets (ad|bc)
            # could be (nquart, 4, nboots+1) to store ALL results...
            idb.create_dataset(
                "quartets", shape=(self.nquartets, 4), dtype=np.uint16)

        # cleanup
        io5.close()
        del tmpseq
        del snpsmap

    def _set_chunksize(self, ncores: int=4):
        """Select a chunksize for parallelizing work.

        For large trees there may be very large number of quartets.
        We will only load part of the list of quartets at a time to
        avoid memory/speed problems. This chunksize is determined by
        the total number and the degree of parallelization.
        """
        breaks = 2
        if self.nquartets < 5000:
            breaks = 1
        if self.nquartets > 100000:
            breaks = 8
        if self.nquartets > 500000:
            breaks = 16
        if self.nquartets > 5000000:
            breaks = 32

        # incorporate n engines into chunksize
        chunk = self.nquartets // (breaks * ncores)
        extra = self.nquartets % (breaks * ncores)
        self._chunksize = max(1, chunk + extra)
        logger.info(f"chunksize: {self._chunksize}")

    def iter_quartet_chunks(self) -> Iterator[Tuple[int, List[Tuple[int,int,int,int]]]]:
        """Yields Lists of quartets (tuples of sample indices).

        The length of yielded lists is chunksize, for sending to
        processors as chunks.
        """
        qiter = itertools.combinations(range(len(self.samples)), 4)

        # yield all ordered quartets in chunks
        if self.nquartets == self.nquartets_total:
            for cidx in range(0, self.nquartets, self._chunksize):
                qrts = itertools.islice(qiter, self._chunksize)
                yield cidx, np.array(list(qrts))

        # yield random quartets in chunks w/o replacement. If using
        # this method nquartets <<< nquartets_total.
        else:
            rand = np.arange(self.nquartets_total, dtype=np.uint32)
            self._rng.shuffle(rand)
            rand = rand[:self.nquartets]
            rand.sort()
            chunk = []
            cidx = 0
            for idx, qrt in enumerate(qiter):
                if idx in rand:
                    chunk.append(qrt)
                if len(chunk) == self._chunksize:
                    yield cidx, np.array(chunk)
                    chunk = []
                    cidx += idx
            if chunk:
                yield cidx, np.array(chunk)

    def run(self, cores: int=0, ipyclient: Optional['ipp.Client']=None):
        """Run quartet inference in parallel...

        """
        cores = max(cores, 1)
        self._set_chunksize(ncores=cores)

        # start a cluster for multiprocessing
        with Cluster(cores=cores) as client:
            lbview = client.load_balanced_view()

            # iter jobs from 0 (orig) to nboots + 1, but starting from
            # the last finished job.
            for jidx in range(0, self.nboots + 1):

                # resample locus dataset (deletes and replaces dsets)
                if jidx:
                    self._resample_database()

                # open the database in writer swmr mode.
                io5 = h5py.File(self.files.database, 'a', libver="latest")
                if not jidx:
                    io5.swmr_mode = True
                resolved_quartets = io5["quartets"]

                # keep track of mean snps per quartet
                sum_snps = 0
                sum_qrts = 0

                # send chunks of quartets to remote engines until all
                # quartets have been processed. Each time one finishes
                # iter and process the next chunk.
                qiter = self.iter_quartet_chunks()
                rasyncs = {-i: lbview.apply(time.sleep, 1e-8) for i in range(cores)}
                while 1:
                    # are there any finished jobs?
                    finished = [i for (i, j) in rasyncs.items() if j.ready()]
                    for key in finished:

                        # store finished result to h5 using swmr
                        result = rasyncs.pop(key).get()
                        if result is not None:  # the null starter jobs
                            rqrts, nsnps, scores = result
                            resolved_quartets[key:key + rqrts.shape[0]] = rqrts
                            resolved_quartets.flush() # not sure if necessary
                            sum_snps += nsnps.sum()
                            sum_qrts += nsnps.size

                        # send a new job to engine
                        try:
                            ridx, qrts = next(qiter)
                            args = self.files.database, qrts, True
                            rasyncs[ridx] = lbview.apply(infer_resolved_quartets, *args)
                            logger.debug(f"starting job:{jidx}, chunk={ridx}")
                        except StopIteration:
                            pass

                    # end...
                    if not rasyncs:
                        break
                    time.sleep(0.05)

                # all quartet subtrees have been inferred. Combine their
                # tmp files into a single qdump file to pass to QMC.
                client.purge_everything()
                logger.info(f"mean SNPs per quartet={sum_snps / sum_qrts:.2f}")
                for i in range(rqrts.shape[0]):
                    logger.debug(f"{rqrts[i]}, {scores[i]:.3f}")
                #logger.info(f"\n{resolved_quartets[:]}\n")
                self._dump_qmc()
                self._run_qmc()

    def _resample_database(self):
        """Samples tmpseq array by re-sampling loci with replacement.

        Sampling uses the spans info to ensure ...there is a single
        SNP per linkage block.
        """
        # open so that we can read and write
        with h5py.File(self.files.database, 'r+') as io5:

            # load the original dataset and span information.
            seqarr = io5['seqarr'][:]
            spans = io5['spans'][:]

            # re-sample locus indices with replacement, and get the
            # size of the new concatenated seqarr made of these loci.
            nloci = spans.shape[0]
            lidxs = self._rng.choice(nloci, nloci, replace=True)

            # fill tmparr and tmpmap with resampled locus sequences
            tmparr, tmpmap = jit_resample(seqarr, spans, lidxs,
                seed=self._rng.integers(2**31))
            # logger.info(seqarr[:, :10])
            # logger.info(tmparr[:, :10])
            # logger.info(tmpmap[tmpmap[:, 0]==0].size)

            # resolve ambiguous bases in resampled data randomly.
            tmparr = jit_resolve_ambigs(tmparr,
                seed=self._rng.integers(2**31))

            # convert CATG bases to matrix indices
            tmparr[tmparr == 65] = 0
            tmparr[tmparr == 67] = 1
            tmparr[tmparr == 71] = 2
            tmparr[tmparr == 84] = 3

            # because its size changed we need to delete and replace
            # the dataset, rather than just replace its values.
            del io5["tmpmap"]
            del io5["tmparr"]
            io5.create_dataset("tmpmap", data=tmpmap)
            io5.create_dataset("tmparr", data=tmparr)

    def _dump_qmc(self):
        """Writes resolved quartets to a txt file for QMC.

        Note that quartets for which NO DATA was present were written
        to results as a random resolution.
        """
        # open the h5 database
        with h5py.File(self.files.database, 'r') as io5:

            # write to out
            with open(self.files.quartets, 'w', encoding="utf-8") as qdump:

                qra = range(0, self.nquartets, self._chunksize)
                for idx in qra:
                    qchunk = io5["quartets"][idx:idx + self._chunksize, :]
                    quarts = ["{},{}|{},{}".format(*i) for i in qchunk]
                    # shuffle b/c QMC says input order can matter,
                    # and format for qmc strict format.
                    self._rng.shuffle(quarts)
                    qdump.write("\n".join(quarts) + "\n")

    def _run_qmc(self):
        """Run quartet max-cut QMC on the quartets qdump file."""
        cmd = [
            str(self._qmc),
            f"qrtt={str(self.files.quartets)}",
            f"otre={str(self.files.tmp)}",
        ]
        logger.debug(f"CMD: {' '.join(cmd)}")

        # run QMC on quartets input
        with subprocess.Popen(
            cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
            ) as proc:
            res = proc.communicate()
            if proc.returncode:
                logger.error(res)
                raise TetradError(f"error in QMC: {res}")

        # parse tmp file written by QMC into a tree and rename tips
        tree = toytree.tree(str(self.files.tmp))
        for idx in range(tree.ntips):
            node = tree.get_nodes(str(idx))[0]
            # logger.debug(f"node={node}, name={node.name}->{self.samples[idx]}")
            node.name = self.samples[idx]

        # convert to newick
        newick = tree.write(dist_formatter=None, internal_labels=None)

        # save the tree to file
        if not self.trees.tree.exists():
            with open(self.trees.tree, 'w', encoding="utf-8") as out:
                out.write(newick)
        else:
            with open(self.trees.boots, 'a', encoding="utf-8") as out:
                out.write(newick + "\n")


# def start_tetrad(
#     name: str,
#     data: str,
#     workdir: str = "analysis-tetrad",
#     nquartets: Optional[int] = None,
#     imap: Dict[str, Sequence[str]] = field(default_factory=dict),
#     resolve_ambigs: bool = False,
#     save_invariants: bool = False,
#     seed: Optional[int] = None,
#     ) -> Tetrad2:
#     """Return an initialized Tetrad object."""

#     tet = Tetrad(name, data, workdir, nquartets, imap)
#     init_seqarray(tet, resolve_ambigs, save_invariants)

# def load_tetrad(
#     json: str,
#     nboots: Optional[int] = None,
#     ) -> Tetrad2:
#     """Return a Tetrad object loaded from an existing JSON file."""


if __name__ == "__main__":

    import toytree
    import ipcoal
    import tetrad
    tetrad.set_log_level("DEBUG")
    ipcoal.set_log_level("DEBUG")

    # TRE = toytree.rtree.unittree(20, treeheight=5e6, seed=123)
    # MOD = ipcoal.Model(TRE, Ne=10000, nsamples=2, seed_trees=1, seed_mutations=1)
    # # MOD.sim_snps(1000, max_alleles=2, min_alleles=2, max_mutations=1)
    # MOD.sim_loci(5000, 300)
    # MOD.apply_missing_mask(coverage=0.9, seed=1)
    # MOD.write_snps_to_hdf5(name="test", outdir="/tmp", diploid=True)
    # DATA = Path("/tmp/test.snps.hdf5")

    TRE = toytree.rtree.unittree(6, treeheight=5e6, seed=123)
    MOD = ipcoal.Model(TRE, Ne=10000, nsamples=2, seed_trees=1, seed_mutations=1)
    MOD.sim_loci(1000, 5)
    MOD.apply_missing_mask(coverage=0.9, seed=1)
    MOD.write_snps_to_hdf5(name="small", outdir="/tmp", diploid=True)
    DATA = Path("/tmp/small.snps.hdf5")

    logger.info(f"{DATA} exists: {DATA.exists()}")
    TET = Tetrad(name='test', workdir="/tmp", data=str(DATA), nquartets=0, nboots=3)

    # for qset in TET.iter_quartet_chunks():
        # print(qset)
    TET.run(cores=1)
    print(TET.trees.tree)
