#!/usr/bin/env python

""" the main CLI for calling tetrad """

from __future__ import print_function, division  # Requires Python 2.7+

import os
import sys
import argparse
# import pkg_resources

import numpy as np
import ipyparallel as ipp


import tetrad
from .tetrad import Tetrad
from .tetrad.parallel import set_cid_and_launch_ipcluster_for_cli, get_client
from .tetrad.utils import TetradError, detect_cpus

__interactive__ = 0


def parse_command_line():
    "Parse CLI arguments"

    ## create the parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_MESSAGE)

    ## get version from ipyrad 
    # tetversion = str(pkg_resources.get_distribution('tetrad'))
    tetversion = tetrad.__version__
    parser.add_argument('-v', '--version', action='version', 
        version="tetrad {}".format(tetversion.split()[1]))

    parser.add_argument('-f', "--force", action='store_true',
        help="force overwrite of existing data")

    parser.add_argument('-s', metavar="seq", dest="seq",
        type=str, default=None,
        help="path to input phylip file (only SNPs)")

    parser.add_argument('-j', metavar='json', dest="json",
        type=str, default=None,
        help="load checkpointed/saved analysis from JSON file.")

    parser.add_argument('-m', metavar="method", dest="method",
        type=str, default="all",
        help="method for sampling quartets (all, random, or equal)")

    parser.add_argument('-q', metavar="nquartets", dest="nquartets",
        type=int, default=0,
        help="number of quartets to sample (if not -m all)")

    parser.add_argument('-b', metavar="boots", dest="boots",
        type=int, default=0,
        help="number of non-parametric bootstrap replicates")

    parser.add_argument('-l', metavar="map_file", dest="map",
        type=str, default=None,
        help="map file of snp linkages (e.g., ipyrad .snps.map)")

    parser.add_argument('-r', metavar="resolve", dest='resolve', 
        type=int, default=1, 
        help="randomly resolve heterozygous sites (default=1)")

    parser.add_argument('-n', metavar="name", dest="name",
        type=str, default="test",
        help="output name prefix (default: 'test')")

    parser.add_argument('-o', metavar="workdir", dest="workdir",
        type=str, default="./analysis-tetrad",
        help="output directory (default: creates ./analysis-tetrad)")

    parser.add_argument('-t', metavar="starting_tree", dest="tree",
        type=str, default=None,
        help="newick file starting tree for equal splits sampling")

    parser.add_argument("-c", metavar="CPUs/cores", dest="cores",
        type=int, default=0,
        help="setting -c improves parallel efficiency with --MPI")

    parser.add_argument("-x", metavar="random_seed", dest="rseed",
        type=int, default=None,
        help="random seed for quartet sampling and/or bootstrapping")    

    parser.add_argument('-d', "--debug", action='store_true',
        help="print lots more info to debugger: ipyrad_log.txt.")

    parser.add_argument("--MPI", action='store_true',
        help="connect to parallel CPUs across multiple nodes")

    parser.add_argument("--invariants", action='store_true',
        help="save a (large) database of all invariants")

    parser.add_argument("--ipcluster", metavar="ipcluster", dest="ipcluster",
        type=str, nargs="?", const="default",
        help="connect to ipcluster profile (default: 'default')")

    ## if no args then return help message
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    ## parse args
    args = parser.parse_args()

    ## RAISE errors right away for some bad argument combinations:
    if args.method not in ["random", "equal", "all"]:
        raise TetradError(
            "method argument (-m) must be one of 'all', 'random', or 'equal'"
        )

    ## required args
    if not any(x in ["seq", "json"] for x in vars(args).keys()):
        print("""
    Bad arguments: tetrad command must include at least one of (-s or -j) 
    """)
        parser.print_help()
        sys.exit(1)

    return args



class CLI:
    def __init__(self):

        self.args = parse_command_line()
        self.data = None
        self.get_data()
        self.set_params()
        ipyclient = self.get_ipyclient()
        self.data.run(force=self.args.force, ipyclient=ipyclient)
           

    def get_data(self):
        # if arg JSON then load existing; if force clear results but keep params
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
                    method=self.args.method, 
                    data=self.args.seq, 
                    resolve=self.args.resolve,
                    mapfile=self.args.map, 
                    guidetree=self.args.tree, 
                    nboots=self.args.boots, 
                    nquartets=self.args.nquartets, 
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
        if self.data.checkpoint.boots:
            print(LOADING_MESSAGE.format(
                self.data.name, self.data.params.method, 
                self.data.checkpoint.boots)
            )


    def get_ipyclient(self):

        # if ipyclient is running (and matched profile) then use that one
        if self.args.ipcluster:
            ipyclient = ipp.Client(profile=self.args.ipcluster)
            self.data._ipcluster["cores"] = len(ipyclient)

        # if not then we need to register and launch an ipcluster instance
        else:
            # set CLI ipcluster terms
            self.ipyclient = None
            self.data._ipcluster["cores"] = (
                self.args.cores if self.args.cores else detect_cpus())
            self.data._ipcluster["engines"] = "Local"
            if self.args.MPI:
                self.data._ipcluster["engines"] = "MPI"
                if not self.args.cores:
                    raise TetradError("must provide -c argument with --MPI")
            
            # register to have a cluster-id with unique name
            set_cid_and_launch_ipcluster_for_cli(self.data) 
            ipyclient = get_client()

        # found running client or started one and then found it.
        return ipyclient       


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
Continuing checkpointed analysis: {}
  sampling method: {}
  bootstrap checkpoint: {}
"""

HELP_MESSAGE = """\
  * Example command-line usage ---------------------------------------------- 

  * Read in sequence/SNP data file, provide linkage, output name, ambig option. 
     tetrad -s data.snps.phy -n test             ## input phylip and give name
     tetrad -s data.snps.phy -l data.snps.map    ## sample one SNP per locus
     tetrad -s data.snps.phy -n noambigs -r 0    ## do not use hetero sites

  * Load saved/checkpointed analysis from '.tet.json' file, or force restart. 
     tetrad -j test.tet.json -b 100         ## continue 'test' until 100 boots
     tetrad -j test.tet.json -b 100 -f      ## force restart of 'test'

  * Sampling modes: 'equal' uses guide tree to sample quartets more efficiently 
     tetrad -s data.snps.phy -m all                       ## sample all quartets
     tetrad -s data.snps.phy -m random -q 1e6 -x 123      ## sample 1M randomly
     tetrad -s data.snps.phy -m equal -q 1e6 -t guide.tre ## sample 1M across tree

  * Connect to N cores on a computer (default without -c arg is to use all avail.)
     tetrad -s data.snps.phy -c 20

  * Start an MPI cluster to connect to nodes across multiple available hosts.
     tetrad -s data.snps.phy --MPI     

  * Connect to a manually started ipcluster instance with default or named profile
     tetrad -s data.snps.phy --ipcluster        ## connects to default profile
     tetrad -s data.snps.phy --ipcluster pname  ## connects to profile='pname'

  * Further documentation: http://ipyrad.readthedocs.io/analysis.html
"""


if __name__ == "__main__": 
    CLI()
