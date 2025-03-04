#!/usr/bin/env python

"""Perform operations on trees.

"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from importlib import metadata
from loguru import logger
from toytree import mtree
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project
from tetrad.src.run_inference import infer_supertree

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
    $ tetrad trees test.json --consensus
    # map splits in quartet data 0 to the best tree
    $ tetrad trees test.json --map -q 0
    # map splits in quartet data 1 to the majority-rule consensus
    $ tetrad trees test.json --map -q 1
    # infer mj-rule tree and map quartet supports onto the tree
    $ tetrad trees test.json --consensus --map -q 1 --weights 1
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
    parser.add_argument("json", type=Path, help="A project JSON file")

    # advanced plotting
    parser.add_argument("--consensus", action="store_true", help="...")
    # parser.add_argument("--map", action="store_true", help="...")
    parser.add_argument("--weights", metavar="int", type=int, default=1, help="weighting strategy for quartet max-cut.")
    parser.add_argument("--idxs", metavar="int", type=int, nargs="*", default=None, help="subselect a quartet result table (default=None=all).")
    # parser.add_argument("-n", "--name", type=str, metavar="str", help="name prefix for output files.")
    # parser.add_argument("-w", "--workdir", type=Path, metavar="path", default=".", help="working directory path.")
    # parser.add_argument("-r", "--random-seed", type=int, metavar="int", help="optional: random number generator seed.")
    # parser.add_argument("-x", "--use_weights", action="store_true", help="optional: use weighted quartets max-cut.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="level", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")
    return parser


def run_trees(args):
    """..."""
    try:
        # load the project
        proj = Project.load_json(args.json)

        # run the supertree inference
        nwks = []
        if args.idxs is None:
            args.idxs = range(proj.bootstrap_idx)
        for idx in args.idxs:
            nwk = infer_supertree(proj, idx=idx, weights=args.weights)
            nwks.append(nwk)

        # infer consensus tree
        mtre = mtree(nwks)
        ctre = mtre.get_consensus_tree()
        print(ctre.write(None))

        # [other things like mapping bootstraps to mj or orig]
        # ...

    except Exception:
        logger.exception("Error during tree conversion")


def main():
    parser = get_parser_trees()
    args = parser.parse_args()
    run_trees(args)


if __name__ == "__main__":
    main()
