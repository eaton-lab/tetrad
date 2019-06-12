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


__interactive__ = 0


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

    parser.add_argument('-s', metavar="snps_file", dest="snps",
        type=str, default=None,
        help="path to input data file (snps.hdf5 from ipyrad)")

    parser.add_argument('-j', metavar='json', dest="json",
        type=str, default=None,
        help="load checkpointed/saved analysis from JSON file.")

    parser.add_argument('-q', metavar="nquartets", dest="nquartets",
        type=float, default=0,
        help="number of quartets to sample (if not -m all)")

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
                rkwargs={"force": self.args.force},
                ipyclient=ipyclient,
                show_cluster=True,
                auto=True,
                )
            pool.wrap_run()
           

    def get_data(self):
        """
        """

        # if arg JSON then load existing; if force clear results.
        if self.args.json:
            self.data = Tetrad(
                name=self.args.name, workdir=self.args.workdir, load=True)
            if self.args.force:
                self.data._refresh()

        # else create a new Tetrad class and JSON
        else:
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
Use the force argument (-f) to overwrite old analysis files, or,
Use the JSON argument (-j {}/{}.tet.json) 
to continue analysis of '{}' from last checkpoint.
"""


LOADING_MESSAGE = """\
Continuing checkpointed analysis from bootstrap replicate: {}
"""

HELP_MESSAGE = """\
  * Example command-line usage ---------------------------------------------- 

  * Read in SNP database file and provide name
     tetrad -s data.snps.hdf5 -n test

  * Load saved/checkpointed analysis
     tetrad -j test.tetrad.json -b 100      # continue 'test' until 100 boots
     tetrad -j test.tetrad.json -b 100 -f   # force restart and run until 100

  * Connect to N cores on a computer
     tetrad -s data.snps.hdf5 -c 20

  * Use MPI to connect to multiple nodes on cluster to find 40 cores
     tetrad -s data.snps.hdf5 --MPI -c 40

  * Connect to a manually started ipcluster instance 
     tetrad -s data.snps.hdf5 --ipcluster        # connects to default profile
     tetrad -s data.snps.hdf5 --ipcluster pname  # connects to profile='pname'

  * Further documentation: http://tetrad.readthedocs.io/analysis.html
"""


def main():
    CLI()

if __name__ == "__main__": 
    main()
