#!/usr/bin/env python

""" 
SVD-quartet like tree inference. Modelled on the following papers:

Chifman, J. and L. Kubatko. 2014. Quartet inference from SNP data under 
the coalescent, Bioinformatics, 30(23): 3317-3324.

Chifman, J. and L. Kubatko. 2015. Identifiability of the unrooted species 
tree topology under the coalescent model with time-reversible substitution 
processes, site-specific rate variation, and invariable sites, Journal of 
Theoretical Biology 374: 35-47
"""

# py2/3 compat
from __future__ import print_function, division
from builtins import range, str

# standard lib
import os
import json
import copy
import itertools

# third party
import h5py
import toytree
import numpy as np
from scipy.special import comb

from .utils import TetradError, Params, Trees, Files
from .jitted import resolve_ambigs, jget_spans
from .parallel import Parallel
from .distributor import Distributor


"""
TODO: 
    - cleanup hdf5 and still able to restart from json by rebuilding arrs
    - linkage samples scaff, dist, or None.
    - 
"""


"""
Parallelization works by creating one sampled SNP data set (bootsarr)
and distributing jobs to engines where each receives a list of quartet sets
to run, and each reads in data from the bootsarr HDF array.

In each bootstrap replicate one SNP is randomly sampled from each locus
for every quartet set. This is important, since it maximizes sampling SNPs
that are actually segregatng in that quartet. This means subsampling SNPs
does not take place until the worker function step.

data/
    - snps [names]
    - snpsmap [columns]
    - genos
    - ref..

idb/
    - seqarr (uint8 snp data)
    - bootsmap (int64 loc/site indices)
    - bootsarr (uint8 cleaned snps)

odb/
    - quartets
    - invariants (group)
      - boot0, boot1, boot2, ...
"""



class Tetrad(object):
    """
    The main tetrad object for saving/loading data and checkpointing with JSON,
    connecting to parallel client, and storing results for easy access.

    Params: 
        name (str): 
            A string to use as prefix for outputs.

        data (str): 
            Three options which contain SNPs and information about 
            linkage of SNPs either from reference mapping or denovo assembly.
            1. A .snps.hdf5 file from ipyrad.
            2. A .vcf file produced by any software (CHROM is used as locus).
            3. A tuple of a (.snps.hdf5, and .snpsmap) files from ipyrad.

        nquartets (int): 
            [optional] def=0=all, else user entered number to sample.

        nboots (int): 
            [optional, def=0] number of non-parametric bootstrap replicates.

        resolve_ambigs (bool): 
            [def=True] Whether to include and randomly resolve ambiguous 
            IUPAC sites in each bootstrap replicate.

        load (bool): 
            If True object is loaded from [workdir]/[name].json for continuing
            from a checkpointed analyis.

        quiet (bool): 
            Suppress progressbar from being printed to stdout.

        save_invariants (bool): 
            Store invariants array to hdf5.

    Functions:
        run()

    Attributes:
        params: optional params can be set after object instantiation
        trees: access trees from object after analysis is finished
        samples: names of samples in the data set
    """
    def __init__(self,
        name, 
        data=None,
        workdir="analysis-tetrad",
        nquartets=0, 
        nboots=0,
        load=False,
        save_invariants=False,
        *args, 
        **kwargs):

        # check additional (hidden) arguments from kwargs.
        self.kwargs = {"initarr": True, "cli": False, "boots_only": False}
        self.kwargs.update(kwargs)

        # are we in the CLI?
        self.quiet = (False if not self.kwargs.get("quiet") else True)
        self._cli = (False if not self.kwargs.get("cli") else True)
        self._boots_only = (False if not self.kwargs.get("boots_only") else True)
        self._spacer = ("  " if self._cli else "")

        # name, sample, and workdir
        self.name = name
        self.samples = []
        self.dirs = os.path.abspath(os.path.expanduser(workdir))
        if not os.path.exists(self.dirs):
            os.mkdir(self.dirs)

        # class storage (checks entries, and makes nice OO style).
        self.params = Params()
        self.files = Files()
        self.trees = Trees()

        # required arguments 
        self.params.nboots = nboots
        self.params.nquartets = nquartets
        self.params.resolve_ambigs = resolve_ambigs
        self.params.save_invariants = save_invariants

        # tree paths
        self.trees.tree = os.path.join(self.dirs, self.name + ".tree")
        self.trees.cons = os.path.join(self.dirs, self.name + ".tree.cons")
        self.trees.boots = os.path.join(self.dirs, self.name + ".tree.boots")        
        self.trees.nhx = os.path.join(self.dirs, self.name + ".tree.nhx")

        # io file paths
        self.files.data = os.path.abspath(os.path.expanduser(data))
        self.files.idb = os.path.join(self.dirs, self.name + ".input.hdf5")
        self.files.odb = os.path.join(self.dirs, self.name + ".output.hdf5")
            
        # load arrays, files, samples from data or existing analysis 
        if load:
            self._load(self.name, self.dirs)
        else:
            # if self.kwargs["initarr"]:
            self._init_seqarray()

        # check input files      
        #self._check_file_handles()

        # default ipcluster information for finding a running Client
        self.ipcluster = {
            "cluster_id": "", 
            "profile": "default",
            "engines": "Local", 
            "quiet": 0, 
            "timeout": 60, 
            "cores": 0, 
            "threads": 2,
            "pids": {},
            }

        # private attributes
        self._checkpoint = 0 
        self._chunksize = 1
        self._tmp = None

        # # self.populations ## if we allow grouping samples
        self._count_quartets()

        # stats is written to os.path.join(self.dirs, self.name+".stats.txt")
        self.stats = Params()
        # self.stats.n_quartets_sampled = self.params.nquartets



    def _check_file_handles(self):
        "check file handles and clear old data unless loaded existing data"

        # if not loading files
        if not (self.files.data and os.path.exists(self.files.data)):
            raise TetradError(
                "must enter a data argument or use load=True")

        # check user supplied paths:
        for sfile in self.files:
            if self.files[sfile]:
                if not os.path.exists(self.files[sfile]):
                    raise TetradError(
                        "file path not found: {}".format(self.files[sfile])
                        )

        # remove any existing results files
        for sfile in self.trees:
            if self.trees[sfile]:
                if os.path.exists(self.trees[sfile]):
                    os.remove(self.trees[sfile])


    def _count_quartets(self):
        """
        Depending on the quartet sampling method selected the number of 
        quartets that must be sampled will be calculated, or error raised.
        """
        total = int(comb(len(self.samples), 4))
        rough = int(len(self.samples) ** 2.8)

        if not self.params.nquartets:
            if rough >= total:
                self.params.nquartets = total
                self._print(
                    "quartet sampler (full): {} / {}"
                    .format(self.params.nquartets, total)
                )
            else:
                self.params.nquartets = rough
                self._print(
                    "quartet sampler (random, nsamples**2.8): {} / {}"
                    .format(self.params.nquartets, total)
                )

        else:
            if self.params.nquartets > total:
                self.params.nquartets = total
                self._print(
                    "quartet sampler (full): {} / {}"
                    .format(self.params.nquartets, total)
                )
            else:                
                self._print(
                    "quartet sampler (random): {} / {}"
                    .format(self.params.nquartets, total)
                )
        # else:
            # self._print(
                # "quartet sampling (random, N**2.8): {} / {}"
                # .format(self.params.nquartets, total))



    def _load_file_paths(self):
        "load file paths if they exist, or None if empty"
        for key, val in self.trees.__dict__.items():
            if not os.path.exists(val):
                self.trees.__dict__[key] = None


    def _init_seqarray(self, quiet=True):
        """ 
        Fills the seqarr with the full SNP data set while keeping memory 
        requirements super low, and creates a bootsarr copy with the 
        following modifications:

        1) converts "-" into "N"s, since they are similarly treated as missing. 
        2) randomly resolve ambiguities (RSKWYM)
        3) convert to uint8 for smaller memory load and faster computation. 
        """
        # get data shape from io5 input file       
        assert ".snps.hdf5" in self.files.data, "data file is not .snps.hdf5"
        io5 = h5py.File(self.files.data, 'r')
        names = [i.decode() for i in io5["snps"].attrs["names"]]
        self.samples = names
        ntaxa = len(names)
        nsnps = io5["snps"].shape[1]
        self._print(
            "loading snps array [{} taxa x {} snps]".format(ntaxa, nsnps))

        # data base file to write the transformed array to
        idb = h5py.File(self.files.idb, 'w')

        # store maps info (enforced to 0-indexed! -- assumes it is not already)
        idb.create_dataset("bootsmap", (nsnps, 2), dtype=np.uint32)
        snpsmap = io5["snpsmap"][:]
        snpsmap[:, 0] = snpsmap[:, 0] - 1
        snpsmap[:, 1] = np.arange(snpsmap.shape[0])
        nloci = np.unique(snpsmap[:, 0]).size
        idb["bootsmap"][:] = snpsmap[:, :2]

        # store spans between loci
        idb.create_dataset("spans", (nloci, 2), dtype=np.int64)
        idb["spans"][:] = jget_spans(snpsmap[:, :2])

        # store snps info
        idb.create_dataset("seqarr", (ntaxa, nsnps), dtype=np.uint8)
        tmpseq = io5["snps"][:].astype(np.uint8)
        tmpseq[tmpseq == 45] = 78
        tmpseq[tmpseq == 95] = 78
        idb["seqarr"][:] = tmpseq

        # boot samples: resolve ambigs and convert CATG bases to matrix indices
        idb.create_dataset("bootsarr", (ntaxa, nsnps), dtype=np.uint8)
        tmpseq = resolve_ambigs(tmpseq)       
        tmpseq[tmpseq == 65] = 0
        tmpseq[tmpseq == 67] = 1
        tmpseq[tmpseq == 71] = 2
        tmpseq[tmpseq == 84] = 3
        idb["bootsarr"][:] = tmpseq

        # report 
        self._print(
            "max unlinked SNPs per quartet (nloci): {}"
            .format(nloci))

        # cleanup
        io5.close()
        idb.close()
        del tmpseq
        del snpsmap


    def _store_N_samples(self, force, ipyclient):
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
        ncpus = len(ipyclient)    
        self._chunksize = max([
            1, 
            sum([
                (self.params.nquartets // (breaks * ncpus)),
                (self.params.nquartets % (breaks * ncpus)),
            ])
        ])

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
        store_all(self)
        # store_random(self)
        # store_equal(self)


    def _refresh(self):
        """ 
        Remove all existing results files and reinit the h5 arrays 
        so that the tetrad object is just like fresh from a CLI start.
        """
        # clear any existing results files
        oldfiles = [
            self.files.qdump, 
            self.files.idb,
            self.files.odb,
        ]
        oldfiles += list(self.trees.values())
        for oldfile in oldfiles:
            if oldfile:
                if os.path.exists(oldfile):
                    os.remove(oldfile)

        # store old ipcluster info
        oldcluster = copy.deepcopy(self.ipcluster)

        # reinit the tetrad object data.
        self.__init__(
            name=self.name, 
            data=self.files.data, 
            workdir=self.dirs,
            resolve_ambigs=self.params.resolve_ambigs,
            save_invariants=self.params.save_invariants,
            nboots=self.params.nboots, 
            nquartets=self.params.nquartets, 
            initarr=True, 
            quiet=True,
            cli=self.kwargs.get("cli"),
            boots_only=self.kwargs.get("boots_only"),
            )

        # retain the same ipcluster info
        self._ipcluster = oldcluster


    def _print(self, message):
        if not self.quiet:
            print(message)


    def run(self, force=False, quiet=False, ipyclient=None, auto=False):
        """
        Parameters
        ----------
        force (bool):
            Overwrite existing results for object with the same name
            and workdir as this one.
        quiet (int):
            0=primt nothing; 1=print progress bars; 2=print pringress
            bars and cluster information.
        ipyclient (ipyparallel.Client object):
            A connected ipyclient object. If ipcluster instance is 
            not running on the default profile then ...
        auto (bool):
            Automatically start and cleanup parallel client.
        """
        # force will refresh (clear) database to be re-filled
        if force:
            self._refresh()

        # update quiet param
        self.quiet = quiet

        # distribute parallel job
        pool = Parallel(
            tool=self,
            rkwargs={"force": force, "boots_only": self._boots_only},
            ipyclient=ipyclient,
            show_cluster=False,
            auto=auto,
            )
        pool.wrap_run()


    def _run(self, force, ipyclient, boots_only):
        """
        Run inside wrapped distributed parallel client.
        """
        # fill the quartet sets array
        self._store_N_samples(force, ipyclient)

        # advance checkpoint to 1 if only running bootstraps
        if self._boots_only:
            if not self._checkpoint:
                self._checkpoint = 1

        # run bootstrap replicates (min 1 b/c the original is 'boot' 0)
        for bidx in range(self._checkpoint, self.params.nboots + 1):

            # distribute parallel jobs; starts from checkpoint; 
            Distributor(self, ipyclient, start=None, quiet=False).run()
            self._checkpoint += 1
            self._print("")

        # map bootstrap support onto the full inference tree
        if self._checkpoint > 1:
            mtre = toytree.mtree(self.trees.boots)
            ctre = mtre.get_consensus_tree()
            ctre.write(self.trees.cons)

        # cleanup
        ipyclient.purge_everything()

        # TODO: remove the .hdf5s ... that would limit continuing...
        # os.remove(self.files.idb)
        # os.remove(self.files.odb)


##################################################
## quartet sampling funcs to fill the database
##################################################

# replace self here with tet to be more clear
def store_all(self):
    """
    Populate array with all possible quartets. This allows us to 
    sample from the total, and also to continue from a checkpoint
    """
    with h5py.File(self.files.idb, 'a') as io5:
        fillsets = io5["quartets"]

        # generator for all quartet sets
        qiter = itertools.combinations(range(len(self.samples)), 4)

        # fill by chunksize at a time
        i = 0
        while i < self.params.nquartets:
            # sample a chunk of the next ordered N set of quartets
            dat = np.array(list(itertools.islice(qiter, self._chunksize)))
            end = min(self.params.nquartets, dat.shape[0] + i)
            fillsets[i:end] = dat[:end - i]
            i += self._chunksize


# not yet updated
def store_random(self):
    """
    Populate array with random quartets sampled from a generator.
    Holding all sets in memory might take a lot, but holding a very
    large list of random numbers for which ones to sample will fit 
    into memory for most reasonable sized sets. So we'll load a 
    list of random numbers in the range of the length of total 
    sets that can be generated, then only keep sets from the set 
    generator if they are in the int list. I did several tests to 
    check that random pairs are as likely as 0 & 1 to come up together
    in a random quartet set. 
    """

    with h5py.File(self.files.idb, 'a') as io5:
        fillsets = io5["quartets"]

        ## set generators
        qiter = itertools.combinations(range(len(self.samples)), 4)
        #rand = np.arange(0, n_choose_k(len(self.samples), 4))
        rand = np.arange(0, int(comb(len(self.samples), 4)))
        np.random.shuffle(rand)
        rslice = rand[:self.params.nquartets]
        rss = np.sort(rslice)
        riter = iter(rss)
        del rand, rslice

        ## print progress update 1 to the engine stdout
        print(self._chunksize)

        ## set to store
        rando = next(riter)
        tmpr = np.zeros((self.params.nquartets, 4), dtype=np.uint16)
        tidx = 0
        while 1:
            try:
                for i, j in enumerate(qiter):
                    if i == rando:
                        tmpr[tidx] = j
                        tidx += 1
                        rando = next(riter)

                    ## print progress bar update to engine stdout
                    if not i % self._chunksize:
                        print(min(i, self.params.nquartets))

            except StopIteration:
                break
        ## store into database
        fillsets[:] = tmpr
        del tmpr


# not yet updated for toytree or py3
def store_equal(self):
    """
    Takes a tetrad class object and populates array with random 
    quartets sampled equally among splits of the tree so that 
    deep splits are not overrepresented relative to rare splits, 
    like those near the tips. 
    """

    with h5py.File(self.files.idb, 'a') as io5:
        fillsets = io5["quartets"]

        # require guidetree
        if not os.path.exists(self.files.tree):
            raise TetradError(
                "To use sampling method 'equal' requires a guidetree")
        tre = toytree.etemini.TreeNode(self.files.tree)
        tre.unroot()
        tre.resolve_polytomy(recursive=True)

        ## randomly sample internals splits
        splits = [([self.samples.index(z.name) for z in i],
                   [self.samples.index(z.name) for z in j]) \
                   for (i, j) in tre.get_edges()]

        ## only keep internal splits, not single tip edges
        splits = [i for i in splits if all([len(j) > 1 for j in i])]

        ## how many min quartets shoudl be equally sampled from each split
        squarts = self.params.nquartets // len(splits)

        ## keep track of how many iterators are saturable.
        saturable = 0

        ## turn each into an iterable split sampler
        ## if the nquartets for that split is small, then sample all, 
        ## if it is big then make it a random sampler for that split.
        qiters = []

        ## iterate over splits sampling quartets evenly
        for idx, split in enumerate(splits):
            ## if small number at this split then sample all possible sets
            ## we will exhaust this quickly and then switch to random for 
            ## the larger splits.
            #total = n_choose_k(len(split[0]), 2) * n_choose_k(len(split[1]), 2)
            total = int(comb(len(split[0]), 2)) * int(comb(len(split[1]), 2))
            if total < squarts * 2:
                qiter = (i + j for (i, j) in itertools.product(
                    itertools.combinations(split[0], 2), 
                    itertools.combinations(split[1], 2)))
                saturable += 1

            ## else create random sampler across that split, this is slower
            ## because it can propose the same split repeatedly and so we 
            ## have to check it against the 'sampled' set.
            else:
                qiter = (random_product(split[0], split[1]) for _ \
                         in range(self.params.nquartets))

            ## store all iterators into a list
            qiters.append((idx, qiter))

        ## create infinite cycler of qiters
        qitercycle = itertools.cycle(qiters)

        ## store visited quartets
        sampled = set()

        ## fill chunksize at a time
        i = 0
        empty = set()
        edge_targeted = 0
        random_targeted = 0

        ## keep filling quartets until nquartets are sampled.
        while i < self.params.nquartets:
            ## grab the next iterator
            cycle, qiter = next(qitercycle)

            ## sample from iterators, store sorted set.
            try:
                qrtsamp = tuple(sorted(next(qiter)))
                if qrtsamp not in sampled:
                    sampled.add(qrtsamp)
                    edge_targeted += 1
                    i += 1
                    ## print progress bar update to engine stdout
                    if not i % self._chunksize:
                        print(min(i, self.params.nquartets))                    

            except StopIteration:
                empty.add(cycle)
                if len(empty) == saturable:
                    break


        ## if array is not full then add random samples
        while i <= self.params.nquartets:
            newset = tuple(sorted(np.random.choice(
                range(len(self.samples)), 4, replace=False)))
            if newset not in sampled:
                sampled.add(newset)
                random_targeted += 1
                i += 1
                ## print progress bar update to engine stdout
                if not i % self._chunksize:
                    print(min(i, self.params.nquartets))

        ## store into database
        print(self.params.nquartets)
        fillsets[:] = np.array(tuple(sampled))
        del sampled



########################################
## some itertools cookbook recipes
########################################

# def n_choose_k(n, k):
#     """ 
#     Get the number of quartets as n-choose-k. This is used in equal 
#     splits to decide whether a split should be exhaustively sampled
#     or randomly sampled. Edges near tips can be exhaustive while highly 
#     nested edges can do with less sampling.
#     """
#     mulfunc = lambda x, y: x * y
#     return int(reduce(mulfunc, (Fraction(n  -i, i+1) for i in range(k)), 1))


def random_combination(nsets, n, k):
    """
    Returns nsets unique random quartet sets sampled from
    n-choose-k without replacement combinations.
    """
    sets = set()
    while len(sets) < nsets:
        newset = tuple(sorted(np.random.choice(n, k, replace=False)))
        sets.add(newset)
    return tuple(sets)


def random_product(iter1, iter2):
    """ 
    Random sampler for equal_splits functions
    """
    iter4 = np.concatenate([
        np.random.choice(iter1, 2, replace=False),
        np.random.choice(iter2, 2, replace=False)
        ])
    return iter4


##########################################
## custom exception messages
##########################################

MIDSTREAM_MESSAGE = """
    loaded object method={}
    cannot change sampling methods midstream
    use force argument to start new run with new method
"""

LOADING_RANDOM = """\
    loading {} random quartet samples to infer a starting tree 
    inferring {} quartet trees
"""

LOADING_STARTER = """\
    loading {} equal-splits quartets from starting tree
"""

NO_SNP_FILE = """\
    Cannot find SNP file. You entered: '{}'. 
"""



###############################################
## Save/Load from JSON operations checkpoint.
###############################################



# def _byteify(data, ignore_dicts=False):
#     """
#     converts unicode to utf-8 when reading in json files
#     """
#     if isinstance(data, str):
#         return data.encode("utf-8")

#     if isinstance(data, list):
#         return [_byteify(item, ignore_dicts=True) for item in data]

#     if isinstance(data, dict) and not ignore_dicts:
#         return {
#             _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
#             for key, value in data.items()
#         }
#     return data


# # TODO: testing
# def _save(self):
#     """
#     Save a JSON serialized tetrad instance to continue from a checkpoint.
#     """
#     # save each attribute as dict
#     fulldict = copy.deepcopy(self.__dict__)

#     # decompose objects to dicts
#     for i, j in fulldict.items():
#         if isinstance(j, (Params, Trees, Files)):
#             fulldict[i] = j.__dict__

#     # dump the dumps
#     fulldumps = json.dumps(
#         fulldict,
#         sort_keys=False, 
#         indent=4, 
#         separators=(",", ":"),
#         )

#     # save to file, make dir if it wasn't made earlier
#     assemblypath = os.path.join(self.dirs, self.name + ".tetrad.json")

#     # protect save from interruption
#     done = 0
#     while not done:
#         try:
#             with open(assemblypath, 'w') as jout:
#                 jout.write(fulldumps)
#             done = 1
#         except (KeyboardInterrupt, SystemExit): 
#             print('.')
#             continue


# # TODO: testing
# def _load(self, name, workdir="analysis-tetrad"):
#     """
#     Load JSON serialized tetrad instance to continue from a checkpoint.
#     Loads name, files, dirs, and database files.
#     """
#     # load the JSON string and try with name + .json
#     path = os.path.join(workdir, name)
#     if not path.endswith(".tet.json"):
#         path += ".tet.json"

#     # expand user
#     path = path.replace("~", os.path.expanduser("~"))

#     # load the json file as a dictionary
#     try:
#         with open(path, 'r') as infile:
#             jdat = json.loads(infile.read(), object_hook=_byteify)
#             fullj = _byteify(jdat, ignore_dicts=True)
#     except IOError:
#         raise TetradError(
#             "Cannot find checkpoint (.tet.json) file at: {}"
#             .format(path))

#     # set old attributes into new tetrad object
#     self.name = fullj["name"]
#     self.files.data = fullj["files"]["data"]
#     # self.files.mapfile = fullj["files"]["mapfile"]        
#     self.dirs = fullj["dirs"]
#     self._init_seqarray()
#     self._parse_names()

#     # fill in the same attributes
#     for key in fullj:
        
#         # fill Params a little different
#         if key in ["files", "params", "database", "trees", "stats", "checkpoint"]:
#             filler = fullj[key]
#             for ikey in filler:
#                 setattr(self.__dict__[key], ikey, fullj[key][ikey])
#                 # self.__dict__[key].__setattr__(ikey, fullj[key][ikey])
#         else:
#             self.__setattr__(key, fullj[key])
