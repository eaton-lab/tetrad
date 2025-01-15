#!/usr/bin/env python

""" 
Summarize results by computing consensus, boot, and quartet tree statistics
"""

import time
import toytree
import h5py


class TreeStats:
    """
    ...
    """
    def __init__(self, tet, ipyclient):
        self.tet = tet
        self.ipyclient = ipyclient
        self.lbview = self.ipyclient.load_balanced_view()
        self.start = (self.tet.start if self.tet.start else time.time())
        self.samples = self.tet.samples


    def run(self):
        self.build_bootstrap_consensus()
        self.get_quartet_stats()
        #self.build_nhx_stats()


    def build_bootstrap_consensus(self):
        "Compute sampling stats and consens trees"   
        # make a consensus from bootstrap reps.
        if self.tet.checkpoint.boots:
            boottrees = toytree.mtree(self.tet.trees.boots)
            self.ctre = boottrees.get_consensus_tree()
            self.ctre.write(self.tet.trees.cons)


    def build_nhx_stats(self):
        "Compute quartet sampling stats, and maybe others"
        ## build stats file and write trees
        qtots = {}
        qsamp = {}
        tots = len(self.ctre)
        totn = set(self.ctre.get_tip_labels())

        ## iterate over node traversal
        for node in self.ctre.treenode.traverse():
            # this is slow, needs to look at every sampled quartet
            # so we send it to be processed on engines
            qtots[node] = self.lbview.apply(get_total, *(tots, node))

            # TODO: error here on pickling...
            qsamp[node] = self.lbview.apply(get_sampled, *(self, totn, node))

        ## wait for jobs to finish (+1 to lenjob is for final progress printer)
        alljobs = list(qtots.values()) + list(qsamp.values())
        printstr = ("calculating stats", "")
        done = 0
        while 1:
            done = [i.ready() for i in alljobs]
            self.tet._progressbar(
                sum(done), len(done), self.tet.start, printstr)
            if all(done):
                break
            time.sleep(0.1)

        ## store results in the tree object
        for node in self.ctre.treenode.traverse():
            total = qtots[node].result()
            sampled = qsamp[node].result()
            node.add_feature("quartets_total", total)
            node.add_feature("quartets_sampled", sampled)
        features = ["quartets_total", "quartets_sampled"]

        ## update final progress
        self._progressbar(0, 0, 0, 0, True)

        ## write tree in NHX format 
        with open(self.tet.trees.nhx, 'w') as outtre:
            outtre.write(self.ctre.tree.write(format=0, features=features))        



#############################################
## Tree and statistics operations
#############################################


def get_total(tots, node):
    """ get total number of quartets possible for a split"""
    if (node.is_leaf() or node.is_root()):
        return 0
    else:
        ## Get counts on down edges. 
        ## How to treat polytomies here?
        if len(node.children) > 2:
            down_r = node.children[0]
            down_l = node.children[1]
            for child in node.children[2:]:
                down_l += child
        else:
            down_r, down_l = node.children
        lendr = sum(1 for i in down_r.iter_leaves())
        lendl = sum(1 for i in down_l.iter_leaves())

        ## get count on up edge sister
        up_r = node.get_sisters()[0]
        lenur = sum(1 for i in up_r.iter_leaves())

        ## everyone else
        lenul = tots - (lendr + lendl + lenur)

        ## return product
        return lendr * lendl * lenur * lenul



def get_sampled(data, totn, node):
    """ get total number of quartets sampled for a split"""
    ## convert tip names to ints
    names = sorted(totn)
    cdict = {name: idx for idx, name in enumerate(names)}
    
    ## skip some nodes
    if (node.is_leaf() or node.is_root()):
        return 0
    else:
        ## get counts on down edges
        if len(node.children) > 2:
            down_r = node.children[0]
            down_l = node.children[1]
            for child in node.children[2:]:
                down_l += child
        else:
            down_r, down_l = node.children

        lendr = set(cdict[i] for i in down_r.get_leaf_names())
        lendl = set(cdict[i] for i in down_l.get_leaf_names())

        ## get count on up edge sister
        up_r = node.get_sisters()[0]
        lenur = set(cdict[i] for i in up_r.get_leaf_names())

        ## everyone else
        lenul = set(cdict[i] for i in totn) - set.union(lendr, lendl, lenur)

    idx = 0
    sampled = 0
    with h5py.File(data.database.output, 'r') as io5:
        end = io5["quartets"].shape[0]
        while 1:
            ## break condition
            if idx >= end:
                break

            ## counts matches
            qrts = io5["quartets"][idx:idx + data._chunksize]
            for qrt in qrts:
                sqrt = set(qrt)
                if all([sqrt.intersection(i) for i in [lendr, lendl, lenur, lenul]]):
                    sampled += 1

            ## increase span
            idx += data._chunksize
    return sampled


    