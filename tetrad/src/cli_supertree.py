#!/usr/bin/env python

"""Infer supertree from resolved quartets in wQMC and print to stdout

- Could add option to output/save the QMC formatted file.
"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from importlib import metadata
from loguru import logger
from toytree import tree
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project
from tetrad.src.run_inference import infer_supertree


VERSION = metadata.version("tetrad")


KWARGS = dict(
    prog="supertree",
    usage="tetrad supertree JSON [options]",
    help="infer a supertree w/ wQMC",
    formatter_class=make_wide(RawDescriptionHelpFormatter),
    description=textwrap.dedent("""
        -------------------------------------------------------------------
        | tetrad supertree
        -------------------------------------------------------------------
        | Infer a supertree from [weighted] quartets in wQMC
        -------------------------------------------------------------------
    """),
    epilog=textwrap.dedent(r"""
    Examples
    --------
    # infer supertree for rep 0 from its resolved quartets
    $ tetrad supertree test.json --idx 0 > tree0.nwk

    # infer supertree for rep 0 using weighted resolved quartets
    $ tetrad supertree test.json --idx 0 --weights 1 > tree0_w1.nwk
    
    # infer supertree for rep 10 and write rooted newick
    $ tetrad supertree test.json --idx 10 -o ~taxon[A,B] > tree0.nwk
    """)
)


def get_parser_supertree(parser: ArgumentParser | None = None, data: Path = None) -> ArgumentParser:
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
    parser.add_argument("-i", "--idx", metavar="int", type=int, help="subselect a quartet result table (default=None=all).")
    parser.add_argument("-w", "--weights", metavar="int", type=int, default=1, help="weighting strategy for quartet max-cut (0, 1, or 2).")
    parser.add_argument("-s", "--min-snps", metavar="int", type=int, default=0, help="min SNPs informing a quartet for inclusion in analysis.")
    parser.add_argument("-r", "--min-ratio", metavar="float", type=float, default=1.0, help="min ratio (best-tree-score / mean(alternatives)) for inclusion in analysis.")
    parser.add_argument("-o", "--outgroup", metavar="str", type=str, default=None, help="outgroup to root tree (e.g., 'taxonA' '~outg[0-9].*').")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="level", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")
    # parser.add_argument("-r", "--random-seed", type=int, metavar="int", help="optional: random number generator seed.")
    # parser.add_argument("--map", action="store_true", help="...")
    return parser


def run_supertree(args):
    """..."""
    try:
        # load the project
        proj = Project.load_json(args.json)
        nwk = infer_supertree(proj=proj, idx=args.idx, weights=args.weights, min_snps=args.min_snps, min_ratio=args.min_ratio)

        # root the tree
        if args.outgroup is not None:
            tre = tree(nwk)
            try:
                tre.root(args.outgroup, inplace=True)
                nwk = tre.write()
            except Exception as _:
                logger.warning("Failed to root tree, returning unrooted result")
                nwk = tre.unroot().write()
        print(nwk)

    except Exception:
        logger.exception("Error during supertree inference")


def main():
    parser = get_parser_supertree()
    args = parser.parse_args()
    run_supertree(args)


if __name__ == "__main__":
    main()
