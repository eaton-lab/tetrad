#!/usr/bin/env python

"Helper classes for tetrad based on similar utils in ipyrad"

import os
import sys
import time
import datetime

from scipy.special import comb


class TetradError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class Params(object):
    """
    Store params for tetrad inference.    
    """
    def __init__(self):
        # default params values
        self._dict = {
            # "method": "all", 
            "nboots": 0, 
            "nquartets": 0,
            "resolve_ambigs": True,
            "save_invariants": False, 
        }
        self._i = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        keys = [i for i in sorted(self._dict.keys())]
        if self._i > len(keys) - 1:
            self._i = 0
            raise StopIteration
        else:
            self._i += 1
            return keys[self._i - 1]

    # non-preferred b/c user can set values directly on dict
    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __repr__(self):
        "return simple representation of dict with ~ shortened for paths"
        _repr = ""
        for key, val in self.items():
            _repr += str((key, val)) + "\n"
        return _repr.strip()

    @property
    def nboots(self):
        return self._dict["nboots"]
    @nboots.setter
    def nboots(self, value):
        self._dict["nboots"] = int(value)

    @property
    def nquartets(self):
        return self._dict["nquartets"]
    @nquartets.setter
    def nquartets(self, value):
        # total possible quartets for this nsample
        total = int(comb(len(self.samples), 4))
        if int(value) > total:
            print("total possible quartets={}".format(total))
        self._dict["nquartets"] = min(int(value), total)

    @property
    def resolve_ambigs(self):
        return self._dict["resolve_ambigs"]
    @resolve_ambigs.setter
    def resolve_ambigs(self, value):
        self._dict["resolve_ambigs"] = bool(value)

    @property
    def save_invariants(self):
        return self._dict["save_invariants"]
    @save_invariants.setter
    def save_invariants(self, value):
        self._dict["save_invariants"] = bool(value)


class Files(object):
    """
    Store and/or load tree files from tetrad runs.
    """
    def __init__(self):

        self._dict = {
            "qdump": None,
            "data": None, 
            "mapfile": None, 
            "guidetree": None, 
            "idatabase": None,
            "odatabase": None,
        }
        self._i = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        keys = [i for i in sorted(self._dict.keys())]
        if self._i > len(keys) - 1:
            self._i = 0
            raise StopIteration
        else:
            self._i += 1
            return keys[self._i - 1]

    # non-preferred b/c user can set values directly on dict
    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __repr__(self):
        "return simple representation of dict with ~ shortened for paths"
        frepr = ""
        maxlen = max([len(i) for i in self.keys()])
        printstr = "{:<" + str(2 + maxlen) + "} {:<20}\n"
        for key, val in self.items():
            val = (" " if not val else val)
            frepr += printstr.format(key, val)
        return frepr.strip()

    @property
    def qdump(self):
        return self._dict["qdump"]
    @property
    def data(self):
        return self._dict["data"]
    @property
    def mapfile(self):
        return self._dict["mapfile"]
    @property
    def guidetree(self):
        return self._dict["guidetree"]
    @property
    def odatabase(self):
        return self._dict["odatabase"]
    @property
    def idatabase(self):
        return self._dict["idatabase"]
    
    @qdump.setter
    def qdump(self, value):
        self._dict["qdump"] = value

    @data.setter
    def data(self, value):
        self._dict["data"] = os.path.abspath(os.path.expanduser(value))

    @mapfile.setter
    def mapfile(self, value):
        self._dict["mapfile"] = os.path.abspath(os.path.expanduser(value))

    @guidetree.setter
    def guidetree(self, value):
        self._dict["guidetree"] = os.path.abspath(os.path.expanduser(value))

    @odatabase.setter
    def odatabase(self, value):
        self._dict["odatabase"] = os.path.abspath(os.path.expanduser(value))

    @idatabase.setter
    def idatabase(self, value):
        self._dict["idatabase"] = os.path.abspath(os.path.expanduser(value))
 

class Trees(object):
    """
    Store and/or load tree files from tetrad runs.
    """
    def __init__(self):

        self._data = {
            "qdump": None,
            "data": None, 
            "mapfile": None, 
            "guidetree": None, 
            "idatabase": None,
            "odatabase": None,
        }
        self._i = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        keys = [i for i in sorted(self._dict.keys())]
        if self._i > len(keys) - 1:
            self._i = 0
            raise StopIteration
        else:
            self._i += 1
            return keys[self._i - 1]

    # non-preferred b/c user can set values directly on dict
    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __repr__(self):
        "return simple representation of dict with ~ shortened for paths"
        _repr = ""
        for key, val in self.items():
            _repr += str((key, val)) + "\n"
        return _repr.strip()        

    @property
    def qdump(self):
        return self._data["qdump"]
    @property
    def data(self):
        return self._data["data"]




    # self.trees = Params()
    # self.trees.tree = os.path.join(self.dirs, self.name + ".tre")
    # self.trees.cons = os.path.join(self.dirs, self.name + ".cons.tre")
    # self.trees.boots = os.path.join(self.dirs, self.name + ".boots.tre")        
    # self.trees.nhx = os.path.join(self.dirs, self.name + ".nhx.tre")


class Progress(object):
    """
    Print pretty progress bar
    """       
    def __init__(self, njobs, finished, start, message, quiet=False):
        self.njobs = njobs
        self.finished = finished

        self.start = time.time()
        self.message = message
        self.progress = 100 * (self.finished / float(self.njobs))

        # build the bar
        hashes = '#' * int(self.progress / 5.)
        nohash = ' ' * int(20 - len(hashes))

        # timestamp
        elapsed = datetime.timedelta(seconds=int(time.time() - start))

        # print to stderr
        if self.kwargs["cli"]:
            print("\r[{}] {:>3}% {} | {:<12} ".format(
                hashes + nohash,
                int(self.progress),
                elapsed,
                message[0],
            ), end="")
        else:
            print("\r[{}] {:>3}% {} | {:<12} ".format(*[
                hashes + nohash,
                int(self.progress),
                elapsed,
                message[0],
            ]), end="")
        sys.stdout.flush()
