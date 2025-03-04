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

# PARALLELIZE.
"""

from pathlib import Path
from math import log
from itertools import product
from loguru import logger
import numpy as np
import toytree
from tetrad.src.schema import Project


def iter_resolved_quartets_table(qrts_file: Path):
    """Generator to yield resolved quartets in wQMC format.
    """
    IDXS = [0, 1, 2, 3]
    with open(qrts_file, 'r') as datain:
        for line in datain:
            values = line.split("\t")
            scores = np.array(sorted(values[4:7]), dtype=np.float64)
            weight = np.mean(sorted(scores)[1:])
            # score = 1. - scores.min() / scores.sum()
            min_score = scores.min()
            # print("****", scores, np.mean(scores[1:]), scores.min(), np.mean(scores[1:]) / scores.min())
            score = 0 if not min_score else np.mean(scores[1:]) / scores.min()
            nsnps = int(values[-1])
            yield tuple(int(values[i]) for i in IDXS), int(values[7]), (nsnps, weight, score)


def get_species_tree_qrt_to_edge_mapping(json_file: Path, newick_file: Path):
    """Generator to yield sorted quartets induced by a species tree edge
    """
    # label tree with tetrad idx labels
    proj = Project.load_json(json_file)
    tree = toytree.tree(newick_file).unroot()
    tidxs = {name: int(tidx) for (tidx, name) in proj.samples.items()}
    for node in tree[:tree.ntips]:
        node.tidx = tidxs[node.name]

    # extract resolved quartets from tree
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


def get_quartet_stats(
    json_file: Path,
    newick_file: Path,
    quartets_files: list[Path],
    min_snps: int = 0,
    min_score_diff: float = 1.25,
) -> toytree.ToyTree:
    """Return a ToyTree with 

    Parameters
    ----------
    sptree: Path
        A newick file of the species tree with tips labeled by numeric
        labels from the tetrad JSON file matching those in the quartets
        file.
    quartets_files: list[Path]
        A list of quartets tabular files produced by tetrad containing
        the resolution and stats for each quartet.
    min_snps: int or None
        An optional minimum number of SNPs that must have informed a
        quartet for it to be used to calculate concordance statistics.
    min_score_diff: float
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
    # get mapping of ordered qrt to resolved qrt and edge ID 
    # {oqrt: (int, Node), ...}
    tree, sdict = get_species_tree_qrt_to_edge_mapping(json_file, newick_file)

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

    # iterate over one or more replicate qrt files
    for qrt_file in quartets_files:
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
                if (score < min_score_diff) or (nsnps < min_snps):
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




if __name__ == "__main__":
    
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # ...
    # tree = toytree.rtree.unittree(10, seed=123)
    JSON = "/home/deren/Documents/tools/tetrad/spp-delphinium/spp.json"
    NWK = "/home/deren/Documents/tools/tetrad/spp-delphinium/spp.best_tree.nwk"
    QRT = Path("/home/deren/Documents/tools/tetrad/spp-delphinium/")
    QRTS = sorted(QRT.glob("spp.quartets_*.tsv"))[:100]

    # ....
    tree = get_quartet_stats(JSON, NWK, QRTS, min_snps=0, min_score_diff=1.1)
    feats = ["QC", "QD", "QI", "QF", "nsnps", "scores", "weights", "conc", "disc1", "disc2"]
    # tree.edge_features = tree.edge_features + set(feats)
    print(tree.get_node_data(feats))
    print(tree.write(features=feats))