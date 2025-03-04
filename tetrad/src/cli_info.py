#!/usr/bin/env python

"""CLI tool to print info about a current project JSON file.

"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from loguru import logger
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project


KWARGS = dict(
    prog="info",
    usage="tetrad info JSON [options]",
    help="print info from project JSON file",
    formatter_class=make_wide(RawDescriptionHelpFormatter),
    description=textwrap.dedent("""
        -------------------------------------------------------------------
        | tetrad info
        -------------------------------------------------------------------
        | Print project summary info
        -------------------------------------------------------------------
    """),
    epilog=textwrap.dedent(r"""
    Examples
    --------
    $ tetrad info test.json
    """)
)


def get_parser_info(parser: ArgumentParser | None = None, data: Path = None) -> ArgumentParser:
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
    parser.add_argument("-s", "--samples", action="store_true", help="show sample names.")
    return parser


def run_info(args):
    """..."""
    try:
        proj = Project.load_json(args.json)
        if not args.samples:
            del proj.samples
        print(proj)
    except Exception:
        logger.exception("Error during run.")


def main():
    parser = get_parser_info()
    args = parser.parse_args()
    run_info(args)


if __name__ == "__main__":
    main()
