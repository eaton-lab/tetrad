#!/usr/bin/env python

""" the main CLI for calling tetrad """

from __future__ import print_function, division

import os
import sys
import argparse
import numpy as np
import ipyparallel as ipp
from pkg_resources import get_distribution

from .tetrad import Tetrad
from .utils import TetradError
from .parallel import Parallel, detect_cpus


__interactive__ = 1


def parse_command_line():
    "Parse CLI arguments"

    # create the parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_MESSAGE)

    # get version from ipyrad 
    parser.add_argument('-v', '--version', action='version', 
        version=str(get_distribution("tetrad")))

    parser.add_argument('-f', "--force", action='store_true',
        help="force overwrite of existing data")

    parser.add_argument('-i', metavar="input file", dest="snps",
        type=str, default=None,
        help="path to input data file (.vcf or .hdf5)")

    # parser.add_argument('-j', metavar='json', dest="json",
    # type=str, default=None,
    # help="load checkpointed/saved analysis from JSON file.")

    parser.add_argument('-q', metavar="nquartets", dest="nquartets",
        type=float, default=0,
        help="number of quartets to sample (default=random N**2.8)")

    parser.add_argument('-b', metavar="boots", dest="boots",
        type=float, default=0,
        help="number of non-parametric bootstrap replicates")

    parser.add_argument('-n', metavar="name", dest="name",
        type=str, default="test",
        help="output name prefix (default: 'test')")

    parser.add_argument('-o', metavar="workdir", dest="workdir",
        type=str, default="./analysis-tetrad",
        help="output directory (default: creates ./analysis-tetrad)")

    parser.add_argument("-c", metavar="cores", dest="cores",
        type=int, default=0,
        help="setting -c improves parallel efficiency with --MPI")

    parser.add_argument("-x", metavar="random_seed", dest="rseed",
        type=int, default=None,
        help="random seed for quartet sampling and/or bootstrapping")    

    parser.add_argument("--invariants", action='store_true',
        help="save a (large) database of all invariants")

    parser.add_argument("--MPI", action='store_true',
        help="connect to parallel CPUs across multiple nodes")

    parser.add_argument("--ipcluster", metavar="profile", dest="ipcluster",
        type=str, nargs="?", const="default",
        help="connect to a running ipcluster instance")

    parser.add_argument("--boots-only", action="store_true", 
        help="only run bootstrap replicate inference")

    # if no args then return help message
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # parse args
    args = parser.parse_args()

    # required args
    if not (("snps" in args) or ("json" in args)):
        sys.exit("""
    Bad arguments: tetrad command must include at least one of (-s or -j) 
    """)

    return args



class CLI:
    def __init__(self):

        self.args = parse_command_line()
        print(HEADER.format(str(get_distribution("tetrad")).split()[-1]))
        self.get_data()
        self.set_params()     

        # if ipyclient is running (and matched profile) then use that one
        if self.args.ipcluster:
            ipyclient = ipp.Client(profile=self.args.ipcluster)
            self.data.ipcluster["cores"] = len(ipyclient)

        # if not then we need to register and launch an ipcluster instance
        else:
            # set CLI ipcluster terms
            ipyclient = None
            self.data.ipcluster["cores"] = (
                self.args.cores if self.args.cores else detect_cpus())
            self.data.ipcluster["engines"] = "Local"
            if self.args.MPI:
                self.data.ipcluster["engines"] = "MPI"
                if not self.args.cores:
                    raise TetradError("must provide -c argument with --MPI")

        # get pool object and start parallel job
        pool = Parallel(
            tool=self.data, 
            rkwargs={"force": self.args.force, "boots_only": self.args.boots_only},
            ipyclient=ipyclient,
            show_cluster=True,
            auto=True,
            )
        pool.wrap_run()
           

    def get_data(self):
        """
        """

        # if arg JSON then load existing; if force clear results.
        # if self.args.json:
            # self.data = Tetrad(
                # name=self.args.name, workdir=self.args.workdir, load=True)
            # if self.args.force:
                # self.data._refresh()

        # else create a new Tetrad class and JSON
        # else:
        if 1:
            # create new JSON path
            newjson = os.path.join(
                self.args.workdir, self.args.name + '.tet.json')

            # if a JSON exists and not arg force then bail out
            print("tetrad instance: {}".format(self.args.name))
            if (os.path.exists(newjson)) and (not self.args.force):
                raise SystemExit(
                    QUARTET_EXISTS
                    .format(self.args.name, self.args.workdir, 
                        self.args.workdir, self.args.name, self.args.name)
                    )
            
            # if JSON doesn't exist, or it exists and arg force then create new
            else:
                self.data = Tetrad(
                    name=self.args.name, 
                    workdir=self.args.workdir, 
                    data=self.args.snps, 
                    nboots=int(self.args.boots), 
                    nquartets=int(self.args.nquartets),
                    cli=True,
                    save_invariants=self.args.invariants,
                    boots_only=self.args.boots_only,
                    )


    def set_params(self):

        # set random seed
        np.random.seed(self.args.rseed)

        # boots can be set either for a new object or loaded JSON to continue it
        if self.args.boots:
            self.data.params.nboots = int(self.args.boots)

        # message about whether we are continuing from existing
        if self.data._checkpoint:
            print(
                LOADING_MESSAGE
                .format(self.data.name, self.data._checkpoint)
            )


## CONSTANTS AND WARNINGS
HEADER = """
-------------------------------------------------------
tetrad [v.{}] 
Quartet inference from phylogenetic invariants
-------------------------------------------------------\
"""


QUARTET_EXISTS = """
Error: tetrad analysis '{}' already exists in {} 
Use the force argument (-f) to overwrite old analysis files, 
or -b X to append additional bootstrap reps up to X.
"""


LOADING_MESSAGE = """\
Continuing checkpointed analysis from bootstrap replicate: {}
"""

HELP_MESSAGE = """\
  * Example command-line usage ---------------------------------------------- 

  * Convert VCF to HDF5 and encode linkage block size.
     tetrad -i data.vcf -o outdir -n data5K -l 5000

  * Convert HDF5 to HDF5 with re-encoded linkage block size.
     tetrad -i data.snps.hdf5 -o outdir -n data5K -l 5000

  * Run tree inference from HDF5 file and save result to "outdir/test.tree"
     tetrad -i data.snps.hdf5 -o outdir -n test

  * Run tree inference from checkpointed analysis: continue until 50 bootstraps
     tetrad -i outdir/test.checkpoint.hdf5 -b 50

  * Run parallel tree inference across 20 cores on a single node/computer.
     tetrad -i data.snps.hdf5 -c 20

  * Run parallel tree inference across multiple nodes on a cluster
     tetrad -i data.snps.hdf5 -c 40 --MPI

  * Run parallel tree inference on a manually started ipcluster (see docs).
     tetrad -i data.snps.hdf5 --ipcluster        # connects to default profile
     tetrad -i data.snps.hdf5 --ipcluster pname  # connects to profile='pname'

  * Further documentation: http://tetrad.readthedocs.io/analysis.html
"""


def main():
    CLI()

if __name__ == "__main__": 
    main()
