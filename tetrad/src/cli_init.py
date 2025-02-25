#!/usr/bin/env python

"""Write JSON file to init a project

"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from importlib import metadata
from loguru import logger
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project
from tetrad.src.write_database import write_database

VERSION = metadata.version("tetrad")


KWARGS = dict(
    prog="init",
    usage="tetrad init DATA [options]",
    help="create a project JSON file to start a tetrad analysis",
    formatter_class=make_wide(RawDescriptionHelpFormatter),
    description=textwrap.dedent("""
        -------------------------------------------------------------------
        | tetrad init
        -------------------------------------------------------------------
        | Create a project JSON file
        -------------------------------------------------------------------
    """),
    epilog=textwrap.dedent(r"""
    Examples
    --------
    $ tetrad init data.vcf -n TEST -w /tmp -q 1000
    $ tetrad init data.snps.hdf5 -n TEST -w /tmp -q 1000
    """)
)


def get_parser_init(parser: ArgumentParser | None = None, data: Path = None) -> ArgumentParser:
    """Return a parser for init.
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
    parser.add_argument("-x", "--weights", type=int, metavar="int", default=1, help="optional: use weight strategy (0=None, 1=default, 2=alt).")
    parser.add_argument("-s", "--subsample-snps", action="store_true", help="optional: sample unlinked SNPs (1 per locus).")    
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="level", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")
    return parser


def run_init(args):
    """..."""
    try:
        proj = Project(version=VERSION, **dict(vars(args)))
        write_database(proj)
        proj.save_json()
        logger.debug(proj)
    except Exception:
        logger.exception("Error during init.")


def main():
    parser = get_parser_init()
    args = parser.parse_args()
    run_init(args)


if __name__ == "__main__":
    main()
