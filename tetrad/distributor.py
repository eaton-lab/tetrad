#!/usr/bin/env python

""" 
Remote functions for Tetrad
"""
import os
import sys
import time

import h5py
import numpy as np
import subprocess as sps

from .utils import TetradError, ProgressBar
from .worker import nworker
from .jitted import resolve_ambigs, jshuffle_cols, jget_shape, jfill_boot
import toytree


# If conda installed then the QMC binary should be in this conda env bin 
PLATFORM = ("Linux" if "linux" in sys.platform else "Mac")
BINARY = "find-cut-{}-64".format(PLATFORM)
QMC = os.path.join(sys.base_prefix, "bin", BINARY)

# if pip+github installed then QMC will be relative to this file
if not os.path.exists(QMC):
    TETPATH = (os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    BINPATH = os.path.join(TETPATH, "bin")
    QMC = os.path.join(os.path.abspath(BINPATH), BINARY)

# check for binary
proc = sps.Popen(["which", QMC], stderr=sps.STDOUT, stdout=sps.PIPE)
res = proc.communicate()[0]
if not res:
    raise TetradError("No QMC binary found: {}".format(QMC))



class Distributor:
    """
    Class with functions distribute jobs on remote engines
    """
    def __init__(self, tet, ipyclient, start=None, quiet=False):
        
        # store params
        self.tet = tet
        self.boot = bool(self.tet._checkpoint)
        
        # parallel and progress bars
        self.start = (start if start else time.time())
        self.quiet = quiet        
        self.ipyclient = ipyclient
        self.lbview = self.ipyclient.load_balanced_view()
        
        # store idxs of jobs to be distributed
        self.jobs = range(0, self.tet.params.nquartets, self.tet._chunksize)

        # print progress
        self.printstr = "inferring full tree   "
        if self.boot:
            self.printstr = (
                "bootstrap inference {}"
                .format(self.tet._checkpoint))


    def run(self):

        # reinit the bootsarr (re-resolve_ambigs)
        if self.boot:
            self.sample_bootseq_array_map()

        # distribute jobs for this bootsarr
        self.send_worker_jobs()

        # write result for this iteration.
        self.dump_qmc()
        self.run_qmc()


    def send_worker_jobs(self):
        """
        Run nworker() on remote engines. 
        """
        # submit jobs distriuted across the cluster.
        asyncs = {}
        for job in self.jobs:
            asyncs[job] = self.lbview.apply(nworker, *(self.tet, job))

        # init progress bar
        prog = ProgressBar(
            njobs=len(asyncs), 
            start=time.time(), 
            message=self.printstr, 
        )

        # catch results as they return and enter to HDF and remove
        while 1:
            # gather finished jobs
            finished = [i for i, j in asyncs.items() if j.ready()]

            # iterate over finished list
            for key in finished:
                rasync = asyncs[key]

                # store result and purge it
                prog.finished += 1
                results = rasync.get()
                self.insert_to_hdf5(key, results)
                del asyncs[key]

            # progress bar update
            prog.update()

            # done is counted on finish, so this means we're done
            if len(asyncs):
                time.sleep(0.1)
            else:
                break


    def sample_bootseq_array_map(self):
        """
        Re-samples loci with replacement to fill the bootarr sampling
        only a single SNP from each locus according to the maparr. 
        """
        with h5py.File(self.tet.files.idb, 'r+') as io5:
            
            # load the original data (seqarr and spans)
            snps = io5["seqarr"][:]
            spans = io5["spans"][:]

            # get size of the new locus re-sampled array
            nloci = spans.shape[0]
            loci = np.random.choice(nloci, nloci)
            arrlen = jget_shape(spans, loci)

            # create a new bootsarr and maparr to fill
            del io5["bootsarr"]
            del io5["bootsmap"]
            newbarr = np.zeros((snps.shape[0], arrlen), dtype=np.uint8)
            newbmap = np.zeros((arrlen, 2), dtype=np.uint32)
            newbmap[:, 1] = np.arange(1, arrlen + 1)
            
            # fill the new arrays            
            tmpseq, tmpmap = jfill_boot(snps, newbarr, newbmap, spans, loci)

            # resolve ambiguous bases randomly.
            tmpseq = resolve_ambigs(tmpseq)

            # convert CATG bases to matrix indices
            tmpseq[tmpseq == 65] = 0
            tmpseq[tmpseq == 67] = 1
            tmpseq[tmpseq == 71] = 2
            tmpseq[tmpseq == 84] = 3

            # store data sets
            io5.create_dataset("bootsmap", data=tmpmap)
            io5.create_dataset("bootsarr", data=tmpseq)           
            # print("resampled bootsarr \n %s", io5["bootsarr"][:, :10])
            # print("resampled bootsmap \n %s", io5["bootsmap"][:10, :])


    def sample_bootseq_array(self):
        """
        Takes the seqarray and re-samples columns and saves to bootsarr.
        """
        with h5py.File(self.tet.database.input, 'r+') as io5:  

            ## load in the seqarr and maparr
            seqarr = io5["seqarr"][:]

            ## resample columns with replacement
            newarr = np.zeros(seqarr.shape, dtype=np.uint8)
            cols = np.random.randint(0, seqarr.shape[1], seqarr.shape[1])
            tmpseq = jshuffle_cols(seqarr, newarr, cols)

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


    def insert_to_hdf5(self, chunk, results):
        """
        Enter resolved quartets and (optional) invariant counts into database
        """
        # two result arrs
        chunksize = self.tet._chunksize
        qrts, invs = results

        # store resolved quartets into db
        io5 = h5py.File(self.tet.files.odb, 'r+')
        io5['quartets'][chunk:chunk + chunksize] = qrts

        # [optional] save count matrices (0-indexed)
        if self.tet.params.save_invariants:

            # create a dataset for boots
            bootkey = "boot{}".format(self.tet._checkpoint)
            io5["invariants"].create_dataset(
                name=bootkey, 
                shape=(self.tet.params.nquartets, 16, 16),
                dtype=np.uint16,
                chunks=(self.tet._chunksize, 16, 16),
            )

            # store counts to database
            pathkey = "invariants/{}".format(bootkey)
            io5[pathkey][chunk:chunk + chunksize] = invs

        # cleanup 
        io5.close()


    def dump_qmc(self):
        """
        Writes the inferred quartet sets from the database to a text 
        file to be used as input for QMC. Quartets that had no information
        available (i.e., no SNPs) were written to the database as 0,0,0,0
        and are excluded here from the output.
        """
        # open the h5 database
        with h5py.File(self.tet.files.odb, 'r') as io5:

            # create an output file for writing
            self.tet.files.qdump = os.path.join(
                self.tet.dirs, 
                self.tet.name + ".quartets.txt")

            # write to out
            with open(self.tet.files.qdump, 'w') as qdump:

                # pull from db and write in chunks
                qra = range(0, self.tet.params.nquartets, self.tet._chunksize)
                for idx in qra:
                    qchunk = io5["quartets"][idx:idx + self.tet._chunksize, :]
                    quarts = [tuple(j) for j in qchunk if np.any(j)]

                    # shuffle and format for qmc
                    np.random.shuffle(quarts)
                    chunk = ["{},{}|{},{}".format(*i) for i in quarts]
                    qdump.write("\n".join(chunk) + "\n")


    def run_qmc(self):
        """
        Runs quartet max-cut QMC on the quartets qdump file.
        """
        # build command
        self._tmp = os.path.join(self.tet.dirs, ".tmptre")
        cmd = [
            QMC, 
            "qrtt={}".format(self.tet.files.qdump),
            "otre={}".format(self._tmp),
        ]

        # run QMC on quartets input
        proc = sps.Popen(cmd, stderr=sps.STDOUT, stdout=sps.PIPE)
        res = proc.communicate()
        if proc.returncode:
            raise TetradError(res[1])

        # parse tmp file written by QMC into a tree and rename tips
        ttre = toytree.tree(self._tmp)
        for tip in ttre.treenode.get_leaves():
            tip.name = self.tet.samples[int(tip.name)]

        # convert to newick
        newick = ttre.write(tree_format=9)      

        # save the tree to file
        if self.boot:
            # self.tet.checkpoint.boots == self.tet.params.nboots
            with open(self.tet.trees.boots, 'a') as outboot:
                outboot.write(newick + "\n")
        else:
            with open(self.tet.trees.tree, 'w') as outtree:
                outtree.write(newick)

        # save the new checkpoint and file paths
        # self.tet._save()
        # TODO: update this to use the HDF5 to store updates, not JSON.


    # def init_new_invariants_array(self):
    #     """
    #     if this is a bootstrap then init a new boot array in the database
    #     max count in this matrix is 65535 (uint16)... 
    #     """      
    #     bootkey = "boot{}".format(self.tet.checkpoint.boots)
       
    #     # 
    #     with h5py.File(self.tet.files.odb, 'r+') as io5:

    #         # name of this data set
    #         if bootkey not in io5["invariants"].keys():

    #             # create a new invariants data set
    #             io5["invariants"].create_dataset(
    #                 name=bootkey, 
    #                 shape=(self.tet.params.nquartets, 16, 16),
    #                 dtype=np.uint16,
    #                 chunks=(self.tet._chunksize, 16, 16),
    #             )