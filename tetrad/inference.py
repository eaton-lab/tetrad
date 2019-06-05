#!/usr/bin/env python

""" 
Remote functions for Tetrad
"""

import os
import time

import h5py
import toytree
import numpy as np
import subprocess as sps  # used to call quartet joining program

from .utils import TetradError
from .jitted import calculate


# class with functions run on the remote engines
class Inference:
    def __init__(self, tet, ipyclient, start=None, quiet=False):
        self.tet = tet
        self.quiet = quiet
        self.boot = bool(self.tet.checkpoint.boots)
        self.start = (start if start else time.time())
        self.ipyclient = ipyclient
        self.lbview = self.ipyclient.load_balanced_view()
        self.jobs = range(
            self.tet.checkpoint.arr, 
            self.tet.params.nquartets, 
            self.tet._chunksize)
        self.printstr = ("initial tree   ", "")
        if self.boot:
            self.printstr = ("bootstrap trees ", "")

    def run(self):
        self.get_snp_array()
        self.init_new_invariants_array()
        self.fill_database()
        self.write_tree()


    def init_new_invariants_array(self):
        "if this is a bootstrap then init a new boot array in the database"
        ## max val is 65535 in here if uint16
        bootkey = "boot{}".format(self.tet.checkpoint.boots)
        with h5py.File(self.tet.database.output, 'r+') as io5:
            if bootkey not in io5["invariants"].keys():
                io5["invariants"].create_dataset(
                    name=bootkey, 
                    shape=(self.tet.params.nquartets, 16, 16),
                    dtype=np.uint16,
                    chunks=(self.tet._chunksize, 16, 16))


    def get_snp_array(self):
        # resample the bootseq array
        if self.tet.files.mapfile:
            self.sample_bootseq_array_map()
        else:
            self.sample_bootseq_array()


    def insert_to_hdf5(self, chunk, results):
        #two result arrs
        chunksize = self.tet._chunksize
        qrts, invs = results

        ## enter into db
        with h5py.File(self.tet.database.output, 'r+') as io5:
            io5['quartets'][chunk:chunk + chunksize] = qrts

            ## entered as 0-indexed !
            if self.tet.params.save_invariants:
                if self.boot:
                    key = "invariants/boot{}".format(self.tet.checkpoint.boots)
                    io5[key][chunk:chunk + chunksize] = invs
                else:
                    io5["invariants/boot0"][chunk:chunk + chunksize] = invs


    def fill_database(self):
        # submit jobs distriuted across the cluster.
        asyncs = {}
        for job in self.jobs:
            asyncs[job] = self.lbview.apply(nworker, *(self.tet, job))

        # wait for jobs to finish, catch results as they return and
        # enter into HDF5 database and delete to keep memory low.
        done = 0
        while 1:
            ## gather finished jobs
            finished = [i for i, j in asyncs.items() if j.ready()]

            ## iterate over finished list
            for key in finished:
                rasync = asyncs[key]
                if rasync.successful():

                    # store result and purge it
                    done += 1
                    results = rasync.result()
                    self.insert_to_hdf5(key, results)
                    del asyncs[key]
                else:
                    raise TetradError(rasync.result())

            # progress bar is different if first vs boot tree
            if not self.quiet:
                if not self.boot:
                    self.tet._progressbar(
                        len(self.jobs), 
                        done, 
                        self.start, 
                        self.printstr,
                    )
                else:
                    self.tet._progressbar(
                        self.tet.params.nboots, 
                        self.tet.checkpoint.boots, 
                        self.start,
                        self.printstr)

            ## done is counted on finish, so this means we're done
            if len(asyncs):
                time.sleep(0.1)
            else:
                break


    def write_tree(self):
        # dump quartets into a text file for QMC
        self.dump_qmc()

        # send to QMC
        if not self.boot:
            self.run_qmc(0)
            self.tet._print("")

        else:
            self.run_qmc(1)
            if self.tet.checkpoint.boots == self.tet.params.nboots:
                self.tet._print("")

        ## reset the checkpoint arr
        #self.tet.checkpoint.arr = 0


    def sample_bootseq_array(self):
        "Takes the seqarray and re-samples columns and saves to bootsarr."
        ## use 'r+' to read and write to existing array. This is super 
        ## similar to what is called in __init__. 
        with h5py.File(self.tet.database.input, 'r+') as io5:  
            ## load in the seqarr and maparr
            seqarr = io5["seqarr"][:]

            ## resample columns with replacement
            newarr = np.zeros(seqarr.shape, dtype=np.uint8)
            cols = np.random.randint(0, seqarr.shape[1], seqarr.shape[1])
            tmpseq = shuffle_cols(seqarr, newarr, cols)

            ## resolve ambiguous bases randomly. We do this each time so that
            ## we get different resolutions.
            if self.tet.params.resolve_ambigs:
                tmpseq = resolve_ambigs(tmpseq)
        
            ## convert CATG bases to matrix indices
            tmpseq[tmpseq == 65] = 0
            tmpseq[tmpseq == 67] = 1
            tmpseq[tmpseq == 71] = 2
            tmpseq[tmpseq == 84] = 3

            ## fill the boot array with a re-sampled phy w/ replacement
            io5["bootsarr"][:] = tmpseq
            del tmpseq    


    def sample_bootseq_array_map(self):
        """
        Re-samples loci with replacement to fill the bootarr sampling
        only a single SNP from each locus according to the maparr. 
        """
        with h5py.File(self.tet.database.input, 'r+') as io5:
            ## load the original data (seqarr and spans)
            seqarr = io5["seqarr"][:]
            spans = io5["spans"][:]            

            ## get size of the new locus re-samples array
            nloci = spans.shape[0]
            loci = np.random.choice(nloci, nloci)
            arrlen = get_shape(spans, loci)

            ## create a new bootsarr and maparr to fill
            del io5["bootsarr"]
            del io5["bootsmap"]
            newbarr = np.zeros((seqarr.shape[0], arrlen), dtype=np.uint8)
            newbmap = np.zeros((arrlen, 2), dtype=np.uint32)
            newbmap[:, 1] = np.arange(1, arrlen + 1)
            
            ## fill the new arrays            
            tmpseq, tmpmap = fill_boot(seqarr, newbarr, newbmap, spans, loci)

            ## resolve ambiguous bases randomly. We do this each time so that
            ## we get different resolutions.
            if self.tet.params.resolve_ambigs:
                tmpseq = resolve_ambigs(tmpseq)

            ## convert CATG bases to matrix indices
            tmpseq[tmpseq == 65] = 0
            tmpseq[tmpseq == 67] = 1
            tmpseq[tmpseq == 71] = 2
            tmpseq[tmpseq == 84] = 3

            ## store data sets
            io5.create_dataset("bootsmap", data=tmpmap)
            io5.create_dataset("bootsarr", data=tmpseq)


    def dump_qmc(self):
        """
        Writes the inferred quartet sets from the database to a text 
        file to be used as input for QMC. Quartets that had no information
        available (i.e., no SNPs) were written to the database as 0,0,0,0
        and are excluded here from the output.
        """
        ## open the h5 database
        with h5py.File(self.tet.database.output, 'r') as io5:

            ## create an output file for writing
            self.tet.files.qdump = os.path.join(
                self.tet.dirs, 
                self.tet.name + ".quartets.txt")
            with open(self.tet.files.qdump, 'w') as qdump:

                ## pull from db
                for idx in range(
                    0, 
                    self.tet.params.nquartets, 
                    self.tet._chunksize):
                    qchunk = io5["quartets"][idx:idx + self.tet._chunksize, :]
                    quarts = [tuple(j) for j in qchunk if np.any(j)]

                    ## shuffle and format for qmc
                    np.random.shuffle(quarts)
                    chunk = ["{},{}|{},{}".format(*i) for i in quarts]
                    qdump.write("\n".join(chunk) + "\n")


    def run_qmc(self, boot):
        "Runs quartet max-cut QMC on the quartets qdump file."
        # build command
        self._tmp = os.path.join(self.tet.dirs, ".tmptre")
        cmd = [
            ip.bins.qmc, 
            "qrtt=" + self.tet.files.qdump, 
            "otre=" + self._tmp,
        ]

        # run QMC on quartets input
        proc = sps.Popen(cmd, stderr=sps.STDOUT, stdout=sps.PIPE)
        res = proc.communicate()
        if proc.returncode:
            raise TetradError(res[1])

        # parse tmp file written by QMC into a tree and rename tips
        ttre = toytree.tree(self._tmp)
        tips = ttre.treenode.get_leaves()
        for tip in tips:
            tip.name = self.tet.samples[int(tip.name)]
        newick = ttre.treenode.write(format=9)      

        # save the tree to file
        if boot:
            with open(self.tet.trees.boots, 'a') as outboot:
                outboot.write(newick + "\n")
        else:
            with open(self.tet.trees.tree, 'w') as outtree:
                outtree.write(newick)

        # save the new checkpoint and file paths
        self.tet._save()




def nworker(data, chunk):
    """
    Worker to distribute work to jit funcs. Wraps everything on an 
    engine to run single-threaded to maximize efficiency for 
    multi-processing.
    """

    ## set the thread limit on the remote engine
    # TODO: is there a way to make this work for non-MKL (e.g., BLAS)?
    # or ideally to work more generally for both? Maybe just try/except, 
    # maybe OPM_NUMTHREADS?
    oldlimit = set_mkl_thread_limit(1)

    ## open seqarray view, the modified arr is in bootstarr
    with h5py.File(data.database.input, 'r') as io5:
        seqview = io5["bootsarr"][:]
        maparr = io5["bootsmap"][:, 0]
        smps = io5["quartets"][chunk:chunk + data._chunksize]

        ## create an N-mask array of all seq cols
        nall_mask = seqview[:] == 78

    ## init arrays to fill with results
    rquartets = np.zeros((smps.shape[0], 4), dtype=np.uint16)
    rinvariants = np.zeros((smps.shape[0], 16, 16), dtype=np.uint16)

    ## fill arrays with results as we compute them. This iterates
    ## over all of the quartet sets in this sample chunk. It would
    ## be nice to have this all numbified.
    for idx in range(smps.shape[0]):
        sidx = smps[idx]
        seqs = seqview[sidx]

        ## these axis calls cannot be numbafied, but I can't 
        ## find a faster way that is JIT compiled, and I've
        ## really, really, really tried. Tried again now that
        ## numba supports axis args for np.sum. Still can't 
        ## get speed improvements by numbifying this loop.
        # TODO: new numba funcs supported, maybe this time...
        nmask = np.any(nall_mask[sidx], axis=0)
        nmask += np.all(seqs == seqs[0], axis=0) 

        ## here are the jitted funcs
        bidx, invar = calculate(seqs, maparr, nmask, TESTS)

        ## store results
        rquartets[idx] = smps[idx][bidx]
        rinvariants[idx] = invar

    # reset thread limit
    set_mkl_thread_limit(oldlimit)

    # return results...
    return rquartets, rinvariants

