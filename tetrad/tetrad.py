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
import sys
import json
import time
import copy
import itertools

# third party
import toytree
import ctypes             # used to throttle MKL threading (todo: OpenBLAS)
import numpy as np
from scipy.special import comb

from .inference import Inference
from .utils import Progress, TetradError, Params, Trees, Files
from .jitted import get_spans
from .parallel import Parallel


# suppress the terrible h5 warning
import warnings
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py



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

        guidetree (str): 
            [optional] a newick file for guiding 'equal' method.

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
        mapfile=None, 
        nquartets=0, 
        nboots=0, 
        resolve_ambigs=True, 
        load=False,
        save_invariants=False,
        # method='all', 
        # guidetree=None, 
        *args, 
        **kwargs):

        # check additional (hidden) arguments from kwargs.
        self.quiet = False
        self.kwargs = {"initarr": True, "cli": False}
        self.kwargs.update(kwargs)

        # are we in the CLI?
        self._cli = (False if not self.kwargs.get("cli") else True)
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
        self.database = {}

        # required arguments 
        # self.params.method = method
        self.params.nboots = nboots
        self.params.nquartets = nquartets
        self.params.resolve_ambigs = resolve_ambigs
        self.params.save_invariants = save_invariants

        # input files
        self.files.data = data
        self.files.mapfile = mapfile
        # self.files.guidetree = guidetree
        self.files.idatabase = os.path.join(
            self.dirs, self.name + ".input.h5")
        self.files.odatabase = os.path.join(
            self.dirs, self.name + ".output.h5")      
        self._init_file_handles(data)  # , mapfile, guidetree)

        # If loading existing object then get .checkpoint and .files
        if load:
            self._load(self.name, self.dirs)

        # If starting new then parse the data file
        else:
            self._parse_names()
            if self.kwargs["initarr"]:
                self._init_seqarray()

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
        self._chunksize = 1
        self._tmp = None

        # self.populations ## if we allow grouping samples
        # (haven't done this yet)
        self._count_quartets()

        # stats is written to os.path.join(self.dirs, self.name+".stats.txt")
        self.stats = Params()
        self.stats.n_quartets_sampled = self.params.nquartets
        #self.stats.avg

        # checkpointing information
        self.checkpoint = Params()
        self.checkpoint.boots = 0
        self.checkpoint.arr = 0


    def _init_file_handles(self):
        "set handles and clear old data unless loaded existing data"

        self.trees = Params()
        self.trees.tree = os.path.join(self.dirs, self.name + ".tre")
        self.trees.cons = os.path.join(self.dirs, self.name + ".cons.tre")
        self.trees.boots = os.path.join(self.dirs, self.name + ".boots.tre")        
        self.trees.nhx = os.path.join(self.dirs, self.name + ".nhx.tre")

        # check user supplied paths:
        for sfile in self.files:
            if self.files[sfile]:
                if not os.path.exists(self.files[sfile]):
                    raise IOError(
                        "file path not found: {}".format(self.files[sfile])
                        )

        # if not loading files
        if not (self.files.data and os.path.exists(self.files.data)):
            raise TetradError(
                "must enter a data argument or use load=True")

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
        if self.params.method == "all":
            self.params.nquartets = total
        else:
            if not self.params.nquartets:
                self.params.nquartets = int(len(self.samples) ** 2.8)
                self._print(
                    "using default setting for 'random' nquartets = N**2.8 ")

            if self.params.nquartets > total:
                self.params.method = "all"
                self._print(
                    " Warning: nquartets > total possible quartets " + 
                    + "({})\n Changing to sampling method='all'".format(total))


    def _load_file_paths(self):
        "load file paths if they exist, or None if empty"
        for key, val in self.trees.__dict__.items():
            if not os.path.exists(val):
                self.trees.__dict__[key] = None


    def _parse_names(self):
        "parse sample names from the sequence file"
        self.samples = []
        with iter(open(self.files.data, 'r')) as infile:
            next(infile)  
            while 1:
                try:
                    self.samples.append(next(infile).split()[0])
                except StopIteration:
                    break


    def _init_seqarray(self, quiet=False):
        """ 
        Fills the seqarr with the full SNP data set while keeping memory 
        requirements super low, and creates a bootsarr copy with the 
        following modifications:

        1) converts "-" into "N"s, since they are similarly treated as missing. 
        2) randomly resolve ambiguities (RSKWYM)
        3) convert to uint8 for smaller memory load and faster computation. 
        """
        # read in the data (seqfile)
        spath = open(self.files.data, 'r')
        line = spath.readline().strip().split()
        ntax = int(line[0])
        nbp = int(line[1])
        tmpseq = np.zeros((ntax, nbp), dtype=np.uint8)
        if not quiet:
            print("loading seq array [{} taxa x {} bp]".format(ntax, nbp))        
    
        ## create array storage for original seqarray, the map used for
        ## subsampling unlinked SNPs (bootsmap) and an array that will be
        ## refilled for each bootstrap replicate (bootsarr).
        with h5py.File(self.database.input, 'w') as io5:
            io5.create_dataset("seqarr", (ntax, nbp), dtype=np.uint8)
            io5.create_dataset("bootsarr", (ntax, nbp), dtype=np.uint8)
            io5.create_dataset("bootsmap", (nbp, 2), dtype=np.uint32)

            # if there is a map file, load it into the bootsmap
            if self.files.mapfile:
                with open(self.files.mapfile, 'r') as inmap:
                    
                    # parse the map file from txt and save as dataset
                    maparr = np.genfromtxt(inmap, dtype=np.uint64)
                    maparr[:, 1] = 0
                    maparr = maparr.astype(int)
                    io5["bootsmap"][:] = maparr[:, [0, 3]]

                    # parse the span info from maparr and save to dataset
                    spans = np.zeros((maparr[-1, 0], 2), dtype=np.uint64)
                    spans = get_spans(maparr, spans)
                    io5.create_dataset("spans", data=spans)
                    if not quiet:
                        print("max unlinked SNPs per quartet (nloci): {}"\
                              .format(spans.shape[0]))
            else:
                io5["bootsmap"][:, 0] = np.arange(io5["bootsmap"].shape[0])

            ## fill the tmp array from the input phy
            for line, seq in enumerate(spath):
                tmpseq[line] = (
                    np.array(list(seq.split()[-1]))
                    .astype(bytes)
                    .view(np.uint8)
                )              

            ## convert '-' or '_' into 'N'
            tmpseq[tmpseq == 45] = 78
            tmpseq[tmpseq == 95] = 78            

            ## save array to disk so it can be easily accessed by slicing
            ## This hardly modified array is used again later for sampling boots
            io5["seqarr"][:] = tmpseq

            ## resolve ambiguous IUPAC codes, this is done differently each rep.
            ## everything below here (resolve, index) is done each rep.
            if self.params.resolve_ambigs:
                tmpseq = resolve_ambigs(tmpseq)

            ## convert CATG bases to matrix indices, nothing else matters.
            tmpseq[tmpseq == 65] = 0
            tmpseq[tmpseq == 67] = 1
            tmpseq[tmpseq == 71] = 2
            tmpseq[tmpseq == 84] = 3

            ## save modified array to disk            
            io5["bootsarr"][:] = tmpseq

        ## cleanup files and mem
        del tmpseq
        spath.close()


    def _refresh(self):
        """ 
        Remove all existing results files and reinit the h5 arrays 
        so that the tetrad object is just like fresh from a CLI start.
        """
        # clear any existing results files
        oldfiles = [self.files.qdump] + \
            list(self.database.__dict__.values()) + \
            list(self.trees.__dict__.values())
        for oldfile in oldfiles:
            if oldfile:
                if os.path.exists(oldfile):
                    os.remove(oldfile)

        # store old ipcluster info
        oldcluster = copy.deepcopy(self._ipcluster)

        # reinit the tetrad object data.
        self.__init__(
            name=self.name, 
            data=self.files.data, 
            mapfile=self.files.mapfile,
            workdir=self.dirs,
            # method=self.params.method,
            # guidetree=self.files.guidetree,
            resolve_ambigs=self.params.resolve_ambigs,
            save_invariants=self.params.save_invariants,
            nboots=self.params.nboots, 
            nquartets=self.params.nquartets, 
            initarr=True, 
            quiet=True,
            cli=self.kwargs.get("cli")
            )

        # retain the same ipcluster info
        self._ipcluster = oldcluster


    def _store_N_samples(self, ipyclient):
        """ 
        Find all quartets of samples and store in a large array
        A chunk size is assigned for sampling from the array of quartets
        based on the number of cpus available. This should be relatively 
        large so that we don't spend a lot of time doing I/O, but small 
        enough that jobs finish often for checkpointing.
        """
        # chunking
        breaks = 2
        if self.params.nquartets < 5000:
            breaks = 1
        if self.params.nquartets > 100000:
            breaks = 8
        if self.params.nquartets > 500000:
            breaks = 16
        if self.params.nquartets > 5000000:
            breaks = 32

        ## chunk up the data
        ncpus = len(ipyclient)    
        self._chunksize = max([
            1, 
            sum([
                (self.params.nquartets // (breaks * ncpus)),
                (self.params.nquartets % (breaks * ncpus)),
            ])
        ])

        ## create h5 OUT empty arrays
        ## 'quartets' stores the inferred quartet relationship (1 x 4)
        ## This can get huge, so we need to choose the dtype wisely. 
        ## the values are simply the index of the taxa, so uint16 is good.
        with h5py.File(self.database.output, 'w') as io5:
            io5.create_dataset(
                name="quartets", 
                shape=(self.params.nquartets, 4), 
                dtype=np.uint16, 
                chunks=(self._chunksize, 4))
            ## group for bootstrap invariant matrices ((16, 16), uint32)
            ## these store the actual matrix counts. dtype uint32 can store
            ## up to 4294967295. More than enough. uint16 max is 65535.
            ## the 0 boot is the original seqarray.
            io5.create_group("invariants")

        ## append to h5 IN array (which has the seqarray, bootsarr, maparr)
        ## and fill it with all of the quartet sets we will ever sample.
        ## the samplign method will vary depending on whether this is random, 
        ## all, or equal splits (in a separate but similar function). 
        with h5py.File(self.database.input, 'a') as io5:
            try:
                io5.create_dataset(
                    name="quartets", 
                    shape=(self.params.nquartets, 4), 
                    dtype=np.uint16, 
                    chunks=(self._chunksize, 4),
                    compression='gzip')
            except RuntimeError:
                raise TetradError(
                    "database file already exists for this analysis, you must"
                  + "run with the force flag to overwrite")
            
        # submit store job to write into self.database.input
        if self.params.method == "all":
            rasync = ipyclient[0].apply(store_all, self)
        elif self.params.method == "random":
            rasync = ipyclient[0].apply(store_random, self)
        elif self.params.method == "equal":
            rasync = ipyclient[0].apply(store_equal, self) 

        ## progress bar 
        printstr = ("generating q-sets", "")
        prog = 0        
        while 1:
            if rasync.stdout:
                prog = int(rasync.stdout.strip().split()[-1])
            self._progressbar(
                self.params.nquartets, prog, self.start, printstr)

            if not rasync.ready():
                time.sleep(0.1)
            else:
                break

        if not rasync.successful():
            raise TetradError(rasync.result())
        self._print("")


    def _print(self, message):
        if not self.quiet:
            print(message)


    def _save(self):
        """
        Save a JSON serialized tetrad instance to continue from a checkpoint.
        """
        ## save each attribute as dict
        fulldict = copy.deepcopy(self.__dict__)
        for i, j in fulldict.items():
            if isinstance(j, Params):
                fulldict[i] = j.__dict__
        fulldumps = json.dumps(
            fulldict,
            sort_keys=False, 
            indent=4, 
            separators=(",", ":"),
            )

        ## save to file, make dir if it wasn't made earlier
        assemblypath = os.path.join(self.dirs, self.name + ".tet.json")
        if not os.path.exists(self.dirs):
            os.mkdir(self.dirs)
    
        ## protect save from interruption
        done = 0
        while not done:
            try:
                with open(assemblypath, 'w') as jout:
                    jout.write(fulldumps)
                done = 1
            except (KeyboardInterrupt, SystemExit): 
                print('.')
                continue


    def _load(self, name, workdir="analysis-tetrad"):
        "Load JSON serialized tetrad instance to continue from a checkpoint."

        ## load the JSON string and try with name+.json
        path = os.path.join(workdir, name)
        if not path.endswith(".tet.json"):
            path += ".tet.json"

        ## expand user
        path = path.replace("~", os.path.expanduser("~"))

        ## load the json file as a dictionary
        try:
            with open(path, 'r') as infile:
                fullj = _byteify(json.loads(infile.read(),
                                object_hook=_byteify), 
                            ignore_dicts=True)
        except IOError:
            raise TetradError("""\
        Cannot find checkpoint (.tet.json) file at: {}""".format(path))

        ## set old attributes into new tetrad object
        self.name = fullj["name"]
        self.files.data = fullj["files"]["data"]
        self.files.mapfile = fullj["files"]["mapfile"]        
        self.dirs = fullj["dirs"]
        self._init_seqarray()
        self._parse_names()

        ## fill in the same attributes
        for key in fullj:
            ## fill Params a little different
            if key in ["files", "params", "database", 
                       "trees", "stats", "checkpoint"]:
                filler = fullj[key]
                for ikey in filler:
                    setattr(self.__dict__[key], ikey, fullj[key][ikey])
                    # self.__dict__[key].__setattr__(ikey, fullj[key][ikey])
            else:
                self.__setattr__(key, fullj[key])



    def run(self, force=False, quiet=False, ipyclient=None):
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
        """
        # force overwrite needs to clear out the HDF5 database
        self.quiet = quiet
        if force:
            self._refresh()

        # print nquartet statement
        self._print(
            "inferring {} quartet tree sets".format(self.params.nquartets))

        # wrap the run in a try statement to ensure we properly shutdown
        try:
            # find and connect to an ipcluster instance given the information
            # in the _ipcluster dict. Connect to running one, or launch new. 
            ipyclient = self._get_parallel(ipyclient)

            # fill the input array with quartets to sample
            self.start = time.time()
            self._store_N_samples(ipyclient)

            # calculate invariants for the full seqarray 
            self.start = time.time()
            if os.path.exists(self.trees.tree):
                print("initial tree already inferred")
            else:
                Inference(self, ipyclient, self.start).run()
                    
            # calculate invariants for each bootstrap rep 
            self.start = time.time()
            if self.params.nboots:
                if self.checkpoint.boots == self.params.nboots:
                    print("{} bootstrap trees already inferred"
                          .format(self.checkpoint.boots))
                else:
                    while self.checkpoint.boots < self.params.nboots:
                        self.checkpoint.boots += 1
                        Inference(self, ipyclient, self.start).run()

            # write output stats
            TreeStats(self, ipyclient).run()

        # handle exceptions so they will be raised after we clean up below
        except KeyboardInterrupt as inst:
            print("\nKeyboard Interrupt by user. Cleaning up...")

        except TetradError as inst:
            print("\nError encountered: {}\n[see trace below]\n".format(inst))
            raise 

        except Exception as inst:
            print("\nException encountered: {}\n[see trace below]\n".format(inst))
            raise

        # close client when done or interrupted
        finally:
            self._cleanup_parallel(ipyclient)



    def _run(self):
        pass



##################################################
## quartet sampling funcs to fill the database
##################################################

# replace self here with tet to be more clear
def store_all(self):
    """
    Populate array with all possible quartets. This allows us to 
    sample from the total, and also to continue from a checkpoint
    """
    with h5py.File(self.database.input, 'a') as io5:
        fillsets = io5["quartets"]

        ## generator for all quartet sets
        qiter = itertools.combinations(range(len(self.samples)), 4)
        i = 0
        while i < self.params.nquartets:
            ## sample a chunk of the next ordered N set of quartets
            dat = np.array(list(itertools.islice(qiter, self._chunksize)))
            end = min(self.params.nquartets, dat.shape[0] + i)
            fillsets[i:end] = dat[:end - i]
            i += self._chunksize

            ## send progress update to stdout on engine
            print(min(i, self.params.nquartets))


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

    with h5py.File(self.database.input, 'a') as io5:
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

    with h5py.File(self.database.input, 'a') as io5:
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


#########################################
## globals
#########################################

## the three indexed resolutions of each quartet
TESTS = np.array([[0, 1, 2, 3], 
                  [0, 2, 1, 3], 
                  [0, 3, 1, 2]], dtype=np.uint8)


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


############################################
## Function to limit threading to single
############################################

def set_mkl_thread_limit(cores):
    """
    set mkl thread limit and return old value so we can reset
    when finished. 
    """
    if "linux" in sys.platform:
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
    else:
        mkl_rt = ctypes.CDLL('libmkl_rt.dylib')
    oldlimit = mkl_rt.mkl_get_max_threads()
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
    return oldlimit



###############################################
## Save/Load from JSON operations checkpoint.
###############################################

def _byteify(data, ignore_dicts=False):
    """
    converts unicode to utf-8 when reading in json files
    """
    if isinstance(data, str):
        return data.encode("utf-8")

    if isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]

    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.items()
        }
    return data
