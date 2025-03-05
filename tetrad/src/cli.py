#!/usr/bin/env python

"""Command line tool of tetrad

Examples
--------
>>> # to init a new job (write JSON args) but not start it
>>> tetrad init -s data.snps.hdf5 -n TEST -i IMAP.txt -w /tmp -l 5000
>>>
>>> # command-line usage: start a new job and start running it
>>> tetrad run /tmp/TEST.json -c 10
>>>
>>> # get summary from JSON file
>>> tetrad ... /tmp/TEST.json --...
"""

from textwrap import dedent
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from importlib import metadata
from tetrad.src.utils import make_wide
from tetrad.src.logger_setup import set_log_level
from tetrad.src.cli_init import get_parser_init, run_init
from tetrad.src.cli_run import get_parser_run, run_run_inference
from tetrad.src.cli_info import get_parser_info, run_info
from tetrad.src.cli_supertree import get_parser_supertree, run_supertree
from tetrad.src.cli_consensus import get_parser_consensus, run_consensus
from tetrad.src.cli_concordance import get_parser_concordance, run_concordance

VERSION = metadata.version("tetrad")

def setup_parsers() -> ArgumentParser:
    """Parse command and subcommand args"""
    parser = ArgumentParser(
        "tetrad", 
        usage="tetrad [subcommand] --help",
        formatter_class=make_wide(RawDescriptionHelpFormatter),
        description=dedent("""
            -----------------------------------------------------
            | %(prog)s: quartet species tree analysis          |
            -----------------------------------------------------

            Examples
            --------
            * init/create a new project JSON file
            $ tetrad init data.vcf -n test -w /tmp
            $ tetrad init data.snps.hdf5 -n test -w /tmp -q 1000 -r 123

            * run tree inference on a project
            $ tetrad run /tmp/test.json -c 10 -b 200
            $ tetrad run /tmp/test.json -c 80 --MPI
            """),
        # epilog=textwrap.dedent(r"""...""")
    )
    parser.add_argument("-v", "--version    ", action="version", version=VERSION)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "EXCEPTION"], metavar="STR", default="INFO", help="stderr logging level (DEBUG, INFO, WARNING, ERROR; default=INFO)")    
    subparsers = parser.add_subparsers(
        prog="%(prog)s", required=True, 
        title="subcommands", dest="subcommand", 
        metavar="--------------", 
        help="-----------------------------------------------------",
    )
    get_parser_init(subparsers)
    get_parser_run(subparsers)
    get_parser_info(subparsers)
    get_parser_supertree(subparsers)
    get_parser_consensus(subparsers)
    get_parser_concordance(subparsers)
    return parser


def main(cmd: str = None) -> int:
    """Command line tool."""
    parser = setup_parsers()
    args = parser.parse_args(cmd.split() if cmd else None)

    # set the logging
    set_log_level(args.log_level)

    # require a subcommand
    if not args.subcommand:
        parser.print_help()
        return 1

    if args.subcommand == "init":
        run_init(args)
        return 0

    if args.subcommand == "run":
        run_run_inference(args)
        return 0
        
    if args.subcommand == "info":
        run_info(args)
        return 0

    if args.subcommand == "supertree":
        run_supertree(args)
        return 0        

    if args.subcommand == "consensus":
        run_consensus(args)
        return 0        

    if args.subcommand == "concordance":
        run_concordance(args)
        return 0        


if __name__ == "__main__":
    main()
