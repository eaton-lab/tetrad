#!/usr/bin/env python

"""Perform operations on trees.

1. Infer a consensus tree.
2. Map bootstrap supports onto a specific tree.
3. Calculate concordance stats on a tree.

"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from importlib import metadata
from loguru import logger
from toytree import mtree, tree
from toytree.infer import get_consensus_tree, get_consensus_features
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project
from tetrad.src.run_inference import infer_supertree

VERSION = metadata.version("tetrad")


KWARGS = dict(
    prog="consensus",
    usage="tetrad consensus JSON [options]",
    help="infer a majority-rule consensus tree",
    formatter_class=make_wide(RawDescriptionHelpFormatter),
    description=textwrap.dedent("""
        -------------------------------------------------------------------
        | tetrad consensus
        -------------------------------------------------------------------
        | Infer a majority-rule consensus tree. This will analyze the
        | resolved quartet TSV files produced by the `run` method. 
        -------------------------------------------------------------------
    """),
    epilog=textwrap.dedent(r"""
    Examples
    --------
    # infer majority-rule consensus from full and bootstrap trees
    $ tetrad consensus test.json > cons.nwk

    # infer majority-rule 50 tree w/ filtering options
    $ tetrad consensus test.json --min-mj 50 --min-snps 1000 > cons.nwk

    # infer majority-rule using different score weighting scheme
    $ tetrad consensus test.json --weights 2 > cons2.nwk

    # infer a supertree for rep 0, and map branch supports onto it.
    $ tetrad consensus test.json --idx 0 > tree0_w_support.nwk
    """)
)


def get_parser_consensus(parser: ArgumentParser | None = None, data: Path = None) -> ArgumentParser:
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
    parser.add_argument("-w", "--weights", metavar="int", type=int, default=1, help="weighting strategy for quartet max-cut.")
    parser.add_argument("-s", "--min-snps", metavar="int", type=int, default=0, help="min SNPs to include quartet in supertree inference.")
    parser.add_argument("-r", "--min-ratio", metavar="float", type=float, default=1.0, help="min ratio (best-tree-score / mean(alternatives)) for inclusion in analysis.")
    parser.add_argument("-t", "--tree", metavar="path", type=Path, default=None, help="nwk tree file on which to map supports instead of inferring consensus.")
    parser.add_argument("-o", "--outgroup", metavar="str", type=str, default=None, help="outgroup to root tree (e.g., 'taxonA' '~outg[0-9].*').")
    parser.add_argument("-c", "--cores", metavar="int", type=int, default=4, help="n cores available for parallel processing.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="level", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")
    # parser.add_argument("-r", "--random-seed", type=int, metavar="int", help="optional: random number generator seed.")
    # parser.add_argument("--map", action="store_true", help="...")
    return parser


def run_consensus(args):
    """..."""
    try:
        # load the project
        proj = Project.load_json(args.json)

        # run the supertree inference on [(re-)weighted] quartet scores
        nwks = []
        with ProcessPoolExecutor(max_workers=args.cores) as pool:
            futures = {
                pool.submit(infer_supertree, proj=proj, idx=idx, weights=args.weights): idx
                for idx in range(proj.bootstrap_idx)
            }
            try:
                for future in as_completed(futures):
                    nwks.append(future.result())
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt received! Stopping all processes.")
                # Cancel all pending tasks
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False)
                raise SystemExit(1)

        # infer consensus tree or parse input tree
        mtre = mtree(nwks)
        if args.tree is not None:
            ctre = tree(args.tree)
            ctre = get_consensus_features(tree, mtre)
        else:
            ctre = get_consensus_tree(mtre)

        # root the tree
        if args.outgroup is not None:
            try:
                ctre.root(args.outgroup, inplace=True)
                nwk = ctre.write()
            except Exception as _:
                logger.warning("Failed to root tree, returning unrooted result")
                nwk = ctre.unroot().write()
        else:
            nwk = ctre.write()
        print(nwk)

    except Exception:
        logger.exception("Error during consensus tree inference")


def main():
    parser = get_parser_consensus()
    args = parser.parse_args()
    run_consensus(args)


if __name__ == "__main__":
    main()
