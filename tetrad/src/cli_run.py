#!/usr/bin/env python

"""Load JSON file to continue tree inference.


"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from loguru import logger
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project
from tetrad.src.run_inference import run_inference


KWARGS = dict(
    prog="run",
    usage="tetrad run JSON [options]",
    help="run a tetrad project to produce additional tree replicates",
    formatter_class=make_wide(RawDescriptionHelpFormatter),
    description=textwrap.dedent("""
        -------------------------------------------------------------------
        | tetrad run
        -------------------------------------------------------------------
        | Run additional replicate tree inferences.
        -------------------------------------------------------------------
    """),
    epilog=textwrap.dedent(r"""
    Examples
    --------
    $ tetrad run TEST.json -c 10 
    """)
)


def get_parser_run(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Return a parser for format-gff tool.
    """
    # create parser or connect as subparser to cli parser
    if parser:
        KWARGS['name'] = KWARGS.pop("prog")
        parser = parser.add_parser(**KWARGS)
    else:
        KWARGS.pop("help")
        parser = ArgumentParser(**KWARGS)

    # add arguments
    parser.add_argument("json", type=Path, help="A project JSON file")

    # advanced plotting
    parser.add_argument("-c", "--cores", type=int, metavar="int", default=0, help="number of parallel cores to use.")
    parser.add_argument("-b", "--boots", type=int, metavar="int", default=0, help="number of bootstrap replicates to run.")
    # parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="level", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")
    return parser


def run_run_inference(args):
    """..."""
    try:
        proj = Project.load_json(args.json)
        run_inference(proj, args.cores, args.boots)
    except Exception:
        logger.exception("Error during run.")


def main():
    parser = get_parser_run()
    args = parser.parse_args()
    run_run_inference(args)


if __name__ == "__main__":
    main()
