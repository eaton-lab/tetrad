#!/usr/bin/env python

"""Calculate quartet concordance scores on a fixed topology

TODO
----
Scores based on quartets:

# how often if the concordant quartet inferred over both discordant quartets
QC: quartet concordance

# are the discordant topology frequencies skewed?
QD: quartet differential

# what proportion of replicates were informative (exceeded likelihood differential)?
QI: quartet informativeness

# when this taxon is sampled how often does it produce a concordant topology?
QF: quartet fidelity

# APPROACH
Run the following functions in order...?
"""

from pathlib import Path
from math import log
from itertools import product
from loguru import logger
import numpy as np
import toytree
from tetrad.src.schema import Project

# the feature names of quartet data/stats calculated
QSTATS = ["QC", "QD", "QI", "QF", "nsnps", "scores", "weights", "conc", "disc1", "disc2"]


def qc(node) -> float:
    """Return QC statistic

    When t0 >>> t1 + t2, QC == 1. When t0 == t1 == t2, QC == 0. When
    t0 < t1 | t2, QC < 0. The range is [-1, 1].
    """
    # how many topologies were found, and is it concordant?
    z = int(np.sum(np.array([node.conc, node.disc1, node.disc2]) > 0))
    if z == 1:
        if node.conc:
            return 1.
        else:
            return -1.
    # calculate discordance for z=2 or 3
    a, b, c = node.conc, node.disc1, node.disc2
    nq = a + b + c
    value = 0.
    for i in (a, b, c):
        if i:
            value += (i / nq) * log(i / nq, z)
    return 1. + value


def qd(node) -> float:
    """Return QD statistic.

    When disc1 == disc2, QD == 1. When disc1 >>> disc2, QD == 0. The
    range is [0, 1]. The QD should be interpreted alongside QC. If 
    there is very little discordance (e.g., QC=0.95) then QD can 
    easily take on extreme values. But if there is a lot of discordance
    (e.g., QC=0.2) then QD is more meaningful.
    """
    if not node.disc1 + node.disc2:
        return 1.
    return 1. - (abs(node.disc1 - node.disc2) / (node.disc1 + node.disc2))


def iter_resolved_quartets_table(qrts_file: Path):
    """Generator to yield resolved quartets in wQMC format.
    """
    IDXS = [0, 1, 2, 3]
    with open(qrts_file, 'r') as datain:
        for line in datain:
            values = line.split("\t")
            nsnps = int(values[-1])
            scores = np.array(sorted(values[4:7]), dtype=np.float64)

            # get weights (i.e., used in wQMC) based on strategy
            weight = np.mean(sorted(scores)[1:])

            # get score (used to include vs exclude a quartet from QMC and here)
            min_score = scores.min()
            # score = 1. - scores.min() / scores.sum()
            # print("****", scores, np.mean(scores[1:]), scores.min(), np.mean(scores[1:]) / scores.min())
            score = 0 if not min_score else np.mean(scores[1:]) / scores.min()

            # generator
            yield tuple(int(values[i]) for i in IDXS), int(values[7]), (nsnps, weight, score)


def prepare_fixed_tree(proj: Project, newick_file: Path) -> tuple[toytree.ToyTree, dict]:
    """Return tidx labeled tree and dict mapping oqrts to resolution and edge.
    """
    # parse tree and unroot
    tree = toytree.tree(newick_file).unroot()

    # Add 'tidx' feature labels to each terminal Node from the project.sample dict
    tidxs = {name: int(tidx) for (tidx, name) in proj.samples.items()}
    for node in tree[:tree.ntips]:
        node.tidx = tidxs[node.name]

    # extract resolved quartets from tree and map to (resolution, edge)
    sdict = {}
    qiter = tree.enum.iter_quadripartitions(feature="tidx", collapse=True)
    for e, q in zip(tree[tree.ntips:-1], qiter):
        # sort by idx by side
        for a, b, c, d in product(*q):
            ordered = tuple(sorted([a, b, c, d]))
            side1, side2 = sorted([(a, b), (c, d)])
            a, b = sorted(side1)
            c, d = sorted(side2)
            res = (a, b, c, d)
            if ordered == res:
                sdict[ordered] = (0, e)
            elif ordered == (res[0], res[2], res[1], res[3]):
                sdict[ordered] = (1, e)
            else:
                sdict[ordered] = (2, e)
    return tree, sdict


def set_quartet_data(tree: toytree.ToyTree, sdict: dict, qrt_file: Path, min_snps: int = 0, min_ratio: float = 1.25) -> toytree.ToyTree:
    """Return a ToyTree with quartet data mapped to the edges.

    Parameters
    ----------
    tree: ToyTree
        A tree tree with "tidx" labels as a feature of tip nodes.
        From 'prepare_fixed_tree'        
    sdict: Dict[tuple, tuple]
        A dict mapping ordered quartets to an int resolution and edge.
        From 'prepare_fixed_tree'
    qrts_file: Path
        A quartets tabular file produced by tetrad containing
        the resolution and stats for each quartet.
    min_snps: int or None
        An optional minimum number of SNPs that must have informed a
        quartet for it to be used to calculate concordance statistics.
    min_ratio: float
        The minimum difference between the best and alternative trees
        of a quartet to consider it an informative quartet. This is
        akin to setting a min delta likelihood in an ML analysis. 
        Quartets that do not meet this or the min_snps criterion are
        excluded when calculating other stats, but inform QI. This uses
        the SVD scores of the three SNP matrices, and is the ratio of
        the mean of two worst scoring / the best (lowest) scoring. It
        can be thought of as "how much better is the best tree than the
        two alternatives". A value of 1.25 mean 25% better; a value of
        2 mean 100% (2X) better.
    """

    # parse tree and set storage containers to nodes
    tree.set_node_data("nqrts", default=0, inplace=True)
    tree.set_node_data("QFc", default=0, inplace=True)
    tree.set_node_data("QFd", default=0, inplace=True)
    tree.set_node_data("conc", default=0, inplace=True)
    tree.set_node_data("disc1", default=0, inplace=True)
    tree.set_node_data("disc2", default=0, inplace=True)
    tree.set_node_data("nu", default=0, inplace=True)
    tree.set_node_data("nsnps", default=[], inplace=True)
    tree.set_node_data("weights", default=[], inplace=True)
    tree.set_node_data("scores", default=[], inplace=True)

    # set sptree resolution index on each node container
    for oqrt, (idx, node) in sdict.items():
        node.nqrts += 1

    # get iterator over the resolved quartets in the rep: (oqrt, int)
    qiter = iter_resolved_quartets_table(qrt_file)

    # iterate over ordered qrt and its int resolution
    for q, rhat, (nsnps, weight, score) in qiter:
        # if this qrt is induced by the species tree
        if q in sdict:
            # get the species tree resolution of qrt and its edge
            r, node = sdict[q]

            # store the nsnps and score
            node.nsnps.append(nsnps)
            node.scores.append(score)
            node.weights.append(weight)

            # exclude quartets from downstream stats if they do not
            # pass the threshold for being informative.
            if (score < min_ratio) or (nsnps < min_snps):
                node.nu += 1
                continue

            # if concordant, store stats
            if rhat == r:
                # set QF counts
                for tip in q:
                    tree[tip].QFc += 1
                # set concordant count
                node.conc += 1
            # if discordant, store stats
            else:
                # set QF counts
                for tip in q:
                    tree[tip].QFd += 1

                # didn't match r=0, found rhat=1 or 2 instead.
                if r == 0:
                    if rhat == 1:
                        node.disc1 += 1
                    else:
                        node.disc2 += 1

                # didn't match r=1, found rhat=0 or 2 instead.
                if r == 1:
                    if rhat == 0:
                        node.disc1 += 1
                    else:
                        node.disc2 += 1

                # didn't match r=2, found rhat=0 or 1 instead.
                if r == 2:
                    if rhat == 0:
                        node.disc1 += 1
                    else:
                        node.disc2 += 1
    # validate?
    # return tree with attached Node data.
    return tree


def set_quartet_stats(trees: list[toytree.ToyTree]) -> toytree.ToyTree:
    """Calculate the quartet concordance stats from Node data.

    """
    # get a single tree from the list
    tree = trees[0].copy()

    # add all other tree's data to this tree.
    if len(trees) > 1:
        for t in trees[1:]:
            for nodex, node in zip(tree, t):
                for feat in QSTATS[4:]:
                    setattr(nodex, feat, getattr(nodex, feat) + getattr(node, feat))
                # nodex.nsnps += node.nsnps
                # nodex.weights += node.weights
                # nodex.scores += node.scores
                # # sum ints
                # nodex.nqrts += node.nqrts
                # nodex.conc += node.conc
                # nodex.disc1 += node.disc1
                # nodex.disc2 += node.disc2
                # nodex.nu += node.nu
                # nodex.QFc += node.QFc
                # nodex.QFd += node.QFd        

    # overwrite as means
    tree.set_node_data("nsnps", {i: np.mean(i.nsnps) for i in tree[tree.ntips:-1]}, default=np.nan, inplace=True)
    tree.set_node_data("weights", {i: np.mean(i.weights) for i in tree[tree.ntips:-1]}, default=np.nan, inplace=True)
    tree.set_node_data("scores", {i: np.mean(i.scores) for i in tree[tree.ntips:-1]}, default=np.nan, inplace=True)

    # print(tree.get_node_data())

    # Quartet concordance: how often is the concordant topo inferred over others?
    tree.set_node_data(
        "QC",
        {i: qc(i) for i in tree[tree.ntips:-1]},
        default=np.nan, inplace=True,
    )

    # Quartet informativeness: what proportion are informative (above cutoff...)
    tree.set_node_data(
        "QI",
        {i: 1 - i.nu / (i.conc + i.disc1 + i.disc2 + i.nu) for i in tree[tree.ntips:-1]},
        default=np.nan, inplace=True,
    )

    # Quartet differential: how asymmetric are the two discordant trees?
    tree.set_node_data(
        "QD",
        {i: qd(i) for i in tree[tree.ntips:-1]},
        default=np.nan, inplace=True,
    )

    # Quartet fidelity: how often it is concordant when this taxon is sampled?
    tree.set_node_data(
        "QF",
        {i: i.QFc / (i.QFc + i.QFd) for i in tree[:tree.ntips]},
        default=np.nan, inplace=True,
    )
    return tree


def run_quartet_concordance(proj: Project, newick_file: Path, qrt_files: list[Path], min_snps: int = 0, min_ratio: float = 1.0) -> toytree.ToyTree:
    """Convenience function to return tree w/ stats saved to Nodes."""

    # parse the input fixed tree
    tree, sdict = prepare_fixed_tree(proj, newick_file)

    # extract and map data from one quartet table
    if isinstance(qrt_files, (Path, str)):
        qrt_files = [qrt_files]
    trees = [set_quartet_data(tree, sdict, q, min_snps, min_ratio) for q in qrt_files]

    # calculate stats
    tree = set_quartet_stats(trees)

    # set quartet data as edge features
    tree.edge_features = tree.edge_features | set(QSTATS) - {"QF"}
    return tree



if __name__ == "__main__":
    
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # ...
    # tree = toytree.rtree.unittree(10, seed=123)
    JSON = "/home/deren/Documents/tools/tetrad/spp-delphinium/spp.json"
    NWK = "/home/deren/Documents/tools/tetrad/spp-delphinium/spp.best_tree.nwk"
    QRT = Path("/home/deren/Documents/tools/tetrad/spp-delphinium/")
    QRTS = sorted(QRT.glob("spp.quartets_*.tsv"))[:3]

    # ....
    proj = Project.load_json(JSON)

    # get mapped results for one rep
    tree0 = run_quartet_concordance(proj, NWK, QRTS[:10], 1000, 1.1)

    print(tree0.get_node_data(QSTATS))
    print(tree0.write(features=QSTATS))
