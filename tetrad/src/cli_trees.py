#!/usr/bin/env python

"""Perform operations on trees.

"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from importlib import metadata
from loguru import logger
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project

VERSION = metadata.version("tetrad")


KWARGS = dict(
    prog="trees",
    usage="tetrad trees JSON [options]",
    help="analyze trees",
    formatter_class=make_wide(RawDescriptionHelpFormatter),
    description=textwrap.dedent("""
        -------------------------------------------------------------------
        | tetrad trees
        -------------------------------------------------------------------
        | Analyze trees
        -------------------------------------------------------------------
    """),
    epilog=textwrap.dedent(r"""
    Examples
    --------
    # infer the majority-rule consensus tree from current bootstraps
    $ tetrad trees test.json consensus 
    # map splits in quartet data 0 to the best tree
    $ tetrad trees test.json map -q 0
    # map splits in quartet data 1 to the majority-rule consensus
    $ tetrad trees test.json map -q 1 
    """)
)


def get_parser_trees(parser: ArgumentParser | None = None, data: Path = None) -> ArgumentParser:
    """Return a parser for trees
    """
    # create parser or connect as subparser to cli parser
    if parser:
        KWARGS['name'] = KWARGS.pop("prog")
        parser = parser.add_parser(**KWARGS)
    else:
        KWARGS.pop("help")
        parser = ArgumentParser(**KWARGS)

    # add arguments
    parser.add_argument("data", type=Path, default=data, help="A VCF or SNPS.HDF5 file")

    # advanced plotting
    parser.add_argument("-n", "--name", type=str, metavar="str", help="name prefix for output files.")
    parser.add_argument("-w", "--workdir", type=Path, metavar="path", default=".", help="working directory path.")
    # parser.add_argument("-i", "--imap", type=Path, metavar="path", help="optional: map of sample names to species.")
    parser.add_argument("-q", "--nquartets", type=float, metavar="int", default=0, help="optional: number of quartets to sample.")
    parser.add_argument("-r", "--random-seed", type=int, metavar="int", help="optional: random number generator seed.")
    parser.add_argument("-s", "--subsample-snps", action="store_true", help="optional: sample unlinked SNPs (1 per locus).")
    parser.add_argument("-x", "--use_weights", action="store_true", help="optional: use weighted quartets max-cut.")
    # parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="level", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")
    return parser
