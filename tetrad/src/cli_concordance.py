#!/usr/bin/env python

"""Measure concordance stats with CLI cmd

"""

import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from importlib import metadata
from loguru import logger
from tetrad.src.utils import make_wide
from tetrad.src.schema import Project
from tetrad.src.concordance import prepare_fixed_tree, set_quartet_data, set_quartet_stats, QSTATS

VERSION = metadata.version("tetrad")


KWARGS = dict(
    prog="concordance",
    usage="tetrad concordance JSON [options]",
    help="measure quartet concordance statistics",
    formatter_class=make_wide(RawDescriptionHelpFormatter),
    description=textwrap.dedent("""
        -------------------------------------------------------------------
        | tetrad concordance
        -------------------------------------------------------------------
        | Measure quartet concordance branch statistics to distinguish
        | support from conflict (e.g., Pease et al. 2018). This will 
        | analyze the resolved quartet TSV files produced by the `run` 
        | method. 
        -------------------------------------------------------------------
    """),
    epilog=textwrap.dedent(r"""
    Examples
    --------
    # map quartet concordance stats onto consensus tree
    $ tetrad concordance test.json -t cons.nwk > cons_qc.nwk

    # map quartet concordance stats onto a different tree
    $ tetrad concordance test.json -t tree0.nwk > tree0_qc.nwk

    # map quartet concordance from filtered set of quartets
    $ tetrad concordance test.json -t cons.nwk -s 1000 -r 1.2 > test.nwk
    """)
)


def get_parser_concordance(parser: ArgumentParser | None = None, data: Path = None) -> ArgumentParser:
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
    parser.add_argument("-t", "--tree", metavar="path", type=Path, default=None, help="nwk tree file on which to map supports instead of inferring consensus.")
    parser.add_argument("-w", "--weights", metavar="int", type=int, default=1, help="weighting strategy for quartet max-cut.")
    parser.add_argument("-s", "--min-snps", metavar="int", type=int, default=0, help="min SNPs to include quartet in supertree inference.")
    parser.add_argument("-r", "--min-ratio", metavar="float", type=float, default=1.0, help="min ratio (best-tree-score / mean(alternatives)) for inclusion in analysis.")
    parser.add_argument("-o", "--outgroup", metavar="str", type=str, default=None, help="outgroup to root tree (e.g., 'taxonA' '~outg[0-9].*').")
    parser.add_argument("-c", "--cores", metavar="int", type=int, default=4, help="n cores available for parallel processing.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="level", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")
    # parser.add_argument("-r", "--random-seed", type=int, metavar="int", help="optional: random number generator seed.")
    return parser


def run_concordance(args):
    """..."""
    try:
        # load the project
        proj = Project.load_json(args.json)

        # get quartets files
        quartets_files = list(proj.qrts_file.parent.glob(f"{proj.name}.quartets*.tsv"))

        # get quartets info
        tree, sdict = prepare_fixed_tree(proj, args.tree)

        # iterate over quartets files
        trees = []
        futures = {}
        with ProcessPoolExecutor(max_workers=args.cores) as pool:
            futures = {
                pool.submit(set_quartet_data, tree, sdict, qrt_file, args.min_snps, args.min_ratio): idx
                for idx, qrt_file in enumerate(quartets_files)
            }
            try:
                for future in as_completed(futures):
                    trees.append(future.result())
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt received! Stopping all processes.")
                # Cancel all pending tasks
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False)
                raise SystemExit(1)

        # get the final tree with all stats
        qtree = set_quartet_stats(trees)

        # root the tree
        if args.outgroup is not None:
            try:
                qtree.root(args.outgroup, inplace=True)
                nwk = qtree.write(features=QSTATS)
            except Exception as _:
                logger.warning("Failed to root tree, returning unrooted result")
                nwk = qtree.unroot().write(features=QSTATS)
        else:
            nwk = qtree.write(features=QSTATS)
        print(nwk)

    except Exception:
        logger.exception("Error during concordance statistic inference")


def main():
    parser = get_parser_concordance()
    args = parser.parse_args()
    run_concordance(args)


if __name__ == "__main__":
    main()
