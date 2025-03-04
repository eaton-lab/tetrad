#!/usr/bin/env python

"""Run tree inference (start or continue)

This is the main analysis module. It starts a cluster connection and
wraps the processing, loads the Projects, samples quartets, processes
them remotely, and writes the results.

The results file is a tab-delimited format with columns:
0. qrt taxon 0
1. qrt taxon 1
2. qrt taxon 2
3. qrt taxon 3
4. SVD score for resolution 0 (12|23)
5. SVD score for resolution 1 (13|24)
6. SVD score for resolution 2 (14|23)
7. quartet resolution index (0, 1, or 2)
8. nsnps 
"""

import sys
from typing import Generator
import time
from pathlib import Path
from itertools import islice
from subprocess import Popen, STDOUT, PIPE, run
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from ipyparallel import Client
from tetrad.src.cluster import Cluster
from tetrad.src.schema import Project, RNGStateModel
from tetrad.src.combinations import iter_chunks_random, iter_chunks_full
from tetrad.src.resolve_quartets import infer_resolved_quartets
from tetrad.jit.resample import jit_resample
from tetrad.jit.resolve_ambigs import jit_resolve_ambigs


TIDXS = np.array([
    [0, 1, 2, 3], 
    [0, 2, 1, 3], 
    [0, 3, 1, 2]], dtype=np.uint8,
)


def get_qmc_bin() -> Path:
    """Return the qmc (find-cut-...) binary"""

    # If conda installed then the QMC binary should be in this conda env bin 
    # platform = "Linux" if "linux" in sys.platform else "Mac"
    # binary = f"find-cut-{platform}-64"
    binary = "max-cut-tree"
    qmc = Path(sys.prefix) / "bin" / binary

    # if pip+github installed then QMC will be relative to this file
    if not qmc.exists():
        tetpath = Path(__file__).absolute().parent.parent.parent
        binpath = tetpath / "bin"
        qmc = binpath.absolute() / binary

    # check for binary
    with Popen(
        ["which", qmc], stderr=STDOUT, stdout=PIPE
        ) as proc:
        res = proc.communicate()[0]
        if not res:
            raise IOError(f"No wQMC binary found: {qmc}")
    return qmc


def get_chunksize(nquartets: int, ncores: int) -> int:
    """Return a chunksize optimized for parallelizing work.

    For large trees there may be very large number of quartets.
    We will only load part of the list of quartets at a time to
    avoid memory/speed problems. This chunksize is determined by
    the total number and the degree of parallelization.
    """
    breaks = 2
    if nquartets < 5000:
        breaks = 1
    if nquartets > 100000:
        breaks = 8
    if nquartets > 500000:
        breaks = 16
    if nquartets > 5000000:
        breaks = 32

    # incorporate n engines into chunksize
    chunk = nquartets // (breaks * ncores)
    extra = nquartets % (breaks * ncores)
    chunksize = max(1, chunk + extra)
    logger.info(f"chunksize: {chunksize}")
    return chunksize


def resample_tmp_database(database: Path, rng: int) -> None:
    """Samples tmpseq array by re-sampling loci with replacement.

    Sampling uses the spans info to ensure ...there is a single
    SNP per linkage block.
    """
    rng = np.random.default_rng(rng)

    # open so that we can read and write
    with h5py.File(database, 'r+') as io5:

        # load the original dataset and span information.
        seqarr = io5['seqarr'][:]
        spans = io5['spans'][:]

        # re-sample locus indices with replacement, and get the
        # size of the new concatenated seqarr made of these loci.
        nloci = spans.shape[0]
        lidxs = rng.choice(nloci, nloci, replace=True)

        # fill tmparr and tmpmap with resampled locus sequences
        tmparr, tmpmap = jit_resample(seqarr, spans, lidxs, seed=rng.integers(2**31))

        # resolve ambiguous bases in resampled data randomly.
        tmparr = jit_resolve_ambigs(tmparr, seed=rng.integers(2**31))

        # debug
        # logger.info(seqarr[:, :10])
        # logger.info(tmparr[:, :10])
        # logger.info(tmpmap[tmpmap[:, 0]==0].size)
        # logger.info(list(io5.keys()))
        # logger.info(io5["tmpmap"][:20])

        # convert CATG bases to matrix indices
        tmparr[tmparr == 65] = 0
        tmparr[tmparr == 67] = 1
        tmparr[tmparr == 71] = 2
        tmparr[tmparr == 84] = 3

        # because its size changed we need to delete and replace
        # the dataset, rather than just replace its values.
        del io5["tmpmap"]
        del io5["tmparr"]
        io5.create_dataset("tmpmap", data=tmpmap)
        io5.create_dataset("tmparr", data=tmparr)


def run_qmc(qmc_in_file: Path, qmc_out_file: Path, use_weights: bool) -> None:
    """Run quartet max-cut QMC on the quartets qdump file."""

    # set the weights usage
    weights = "off"
    if use_weights:
        weights = "on"

    # get the wQMC binary
    qmc = get_qmc_bin()

    # build the command
    cmd = [str(qmc), f"qrtt={qmc_in_file}", f"otre={qmc_out_file}", f"weights={weights}"]
    logger.debug(f"CMD: {' '.join(cmd)}")

    # run QMC on quartets input
    with Popen(cmd, stderr=STDOUT, stdout=PIPE) as proc:
        res = proc.communicate()
        if proc.returncode:
            logger.error(res)
            raise Exception(f"error in wQMC: {res}")


def relabel_tree(newick: str, samples: dict[int, str]) -> str:
    """Return the tree with numeric labels replaced with string names.
    """
    import toytree
    # parse tmp file written by QMC into a tree and rename tips
    tree = toytree.tree(newick)
    for idx in range(tree.ntips):
        node = tree.get_nodes(str(idx))[0]
        node.name = samples[idx]

    # convert to newick
    newick = tree.write(dist_formatter=None, internal_labels=None)
    return newick


def distributor(database_file: Path, qrts_file: Path, nsamples: int, qiter: Generator, subsample_snps: bool, client: Client):
    """Returns the completed results for one quartet iterator.

    Parameters
    ----------
    database_file: Path
        The project snps.hdf5 database file.
    qrt_file: Path
        A file path to write resolved quartets to.
    nsamples: int
        The number of samples in the project.
    qiter: Generator
        A generator of quartet chunks (e.g., iter_chunks_random())
    subsample_snps: bool
        If True subsample 1 SNP per locus/scaff.
    client: Client
        A connected ipyparallel Client.
    """
    # load-balancer
    lbview = client.load_balanced_view()

    # store asynchronous results
    rasyncs = {}

    # clear qrt output file to write to.
    qrts_file.write_text("")

    # track jobs and send new ones when a chunk finishes
    qiter = enumerate(qiter)
    while qiter:

        # start new jobs if we haven't reached the limit
        while len(rasyncs) < len(client):
            try:
                jidx, chunk = next(qiter)
                args = (database_file, nsamples, chunk, subsample_snps)
                job = lbview.apply_async(infer_resolved_quartets, *args)
                rasyncs[jidx] = job
            except StopIteration:
                break

        # Check for completed jobs
        strdata = {}
        for jid, job in rasyncs.items():
            if job.ready():
                # retrieve result and write to qrt file
                # TODO: async
                try:
                    rqrts, rstat, rscor = job.get()
                    tabular = pd.concat([pd.DataFrame(i) for i in (rqrts, rscor, rstat)], axis=1)
                    strdata[jid] = tabular.to_csv(sep="\t", float_format='%.6f', index=False, header=False)
                except Exception as e:
                    logger.error(f"Job failed with exception: {e}")
                    raise e

        # write any completed results to file
        if strdata:
            with open(qrts_file, "a") as out:
                for jid, data in strdata.items():
                    out.write(data)
                    del rasyncs[jid]
            strdata = {}
            if not rasyncs:
                break

        # Wait before checking again to avoid high CPU usage
        time.sleep(0.1)
    return qrts_file


def iter_qmc_formatted(qrts_file: Path, weights: int):
    """Generator to yield resolved quartets in wQMC format.
    """
    with open(qrts_file, 'r') as datain:
        for line in datain:
            values = line.split("\t")
            order = int(values[7])
            if order == 1:
                qrts = values[0], values[2], values[1], values[3]
            elif order == 2:
                qrts = values[0], values[3], values[1], values[2]
            else:
                qrts = values[:4]
            scores = np.array(values[4:7], dtype=np.float64)

            # calculate weights
            if not weights:
                weight = 1.0
            elif weights == 1:
                weight = np.mean(sorted(scores)[1:])
            elif weights == 2:
                weight = 1. - scores.min() / scores.sum()
            else:
                raise ValueError(f"no weight strategy {weights}")

            # return formatted for QMC
            yield "{},{}|{},{}:{:.5f}".format(*qrts, weight)


def write_qmc_format(qrts_file: Path, qmc_in_file: Path, weights: int = 0):
    """Write resolved weighted quartets in random order to a tmp file.
    """
    chunk_size = 50_000

    # write weighted quartets to 
    qiter = iter_qmc_formatted(qrts_file, weights)
    with open(qmc_in_file, 'w') as out:
        while 1:
            chunk = "\n".join(islice(qiter, chunk_size))
            if chunk:
                out.write(chunk + "\n")
            else:
                break

    # randomize input order (shuf cannot take a seed?) Maybe just
    # use python instead then...?
    fpath = str(qmc_in_file)
    cmd = ["shuf", "-o", fpath, fpath]
    run(cmd, check=True)  # subprocess


def infer_supertree(proj: Project, idx: int, weights: int) -> str:
    """Return a quartet supertree.
    """
    # get the quartets file for the selected rep
    qrts_file = proj.workdir / f"{proj.name}.quartets_{idx}.tsv"
    # calculate weights and format for QMC
    write_qmc_format(qrts_file, proj.qmc_in_file, weights)
    # run QMC to get supertree
    run_qmc(proj.qmc_in_file, proj.qmc_out_file, bool(weights))
    # relabel tree w/ tip names
    nwk = relabel_tree(proj.qmc_out_file, proj.samples)
    return nwk


def run_inference(proj: Project, ncores: int, nboots: int) -> None:
    """
    """
    # get chunksize given nqrts and num cpus
    chunksize = get_chunksize(proj.nqrts, ncores)

    # start cluster and process chunks
    with Cluster(cores=ncores) as client:

        # create or load the rng
        if proj.bootstrap_rng is None:
            logger.info("starting quartet tree inference")
            rng = np.random.default_rng(proj.random_seed)
        else:
            logger.info("continuing quartet tree inference")        
            rng = proj.bootstrap_rng.to_rng()

        # continue processing until 
        while proj.bootstrap_idx <= nboots:

            # get full or random quartet sampler
            if proj.nqrts == proj.nqrts_total:
                qiter = iter_chunks_full(proj.nsamples, chunksize)
            else:
                qiter = iter_chunks_random(proj.nsamples, proj.nqrts, chunksize, rng)

            # bootstrap resample
            if proj.bootstrap_idx:
                resample_tmp_database(proj.database_file, rng)

            # infer quartets and write to file
            args = (proj.database_file, proj.qrts_file, proj.nsamples, qiter, proj.subsample_snps, client)
            distributor(*args)

            # infer supertree from qrts and write to best or boots
            nwk = infer_supertree(proj, proj.bootstrap_idx, proj.weights)
            if proj.bootstrap_idx == 0:
                with open(proj.best_file, "w") as out:
                    out.write(nwk + "\n")
            else:
                with open(proj.boots_file, "a") as out:
                    out.write(nwk + "\n")

            # advance counter, store RNG,...
            logger.info(f"finished rep {proj.bootstrap_idx}")
            proj.bootstrap_idx += 1
            proj.bootstrap_rng = RNGStateModel.from_rng(rng)
            proj.save_json()


if __name__ == "__main__":

    from tetrad.logger_setup import set_log_level
    set_log_level("INFO")

    # proj = Project.load_json("../../testdir/TEST.json")
    proj = Project.load_json("../../spp3-delphinium/spp3.json")
    # run_inference(proj, 7, 50)
