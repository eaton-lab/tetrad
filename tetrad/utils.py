#!/usr/bin/env python

"Helper classes for tetrad based on similar utils in ipyrad"

import os
import sys
import time
import datetime

import numpy as np


class TetradError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class Files(object):
    """
    Store and/or load tree files from tetrad runs.
    """
    def __init__(self):
        self._dict = {
            "qdump": None,
            "data": None, 
            "idb": None,
            "odb": None,
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

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

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
    def odb(self):
        return self._dict["odb"]
    @property
    def idb(self):
        return self._dict["idb"]
    
    @qdump.setter
    def qdump(self, value):
        self._dict["qdump"] = os.path.abspath(os.path.expanduser(value))
    @data.setter
    def data(self, value):
        self._dict["data"] = os.path.abspath(os.path.expanduser(value))
    @odb.setter
    def odb(self, value):
        self._dict["odb"] = os.path.abspath(os.path.expanduser(value))
    @idb.setter
    def idb(self, value):
        self._dict["idb"] = os.path.abspath(os.path.expanduser(value))
 

class Params(object):
    """
    Store params for tetrad inference.    
    """
    def __init__(self):
        # default params values
        self._dict = {
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

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

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
        self._dict["nquartets"] = value

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


class Trees(object):
    """
    Store and/or load tree files from tetrad runs.
    """
    def __init__(self):

        self._dict = {
            "tree": None,
            "cons": None,
            "boots": None,
            "nhx": None,
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

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value      

    def __repr__(self):
        "return simple representation of dict with ~ shortened for paths"
        _repr = ""
        for key, val in self.items():
            _repr += str((key, val)) + "\n"
        return _repr.strip()        

    @property
    def tree(self):
        return self._dict["tree"]
    @property
    def cons(self):
        return self._dict["cons"]
    @property
    def boots(self):
        return self._dict["boots"]
    @property
    def nhx(self):
        return self._dict["nhx"]

    @tree.setter
    def tree(self, value):
        self._dict["tree"] = value
    @cons.setter
    def cons(self, value):
        self._dict["cons"] = value
    @boots.setter
    def boots(self, value):
        self._dict["boots"] = value
    @nhx.setter
    def nhx(self, value):
        self._dict["nhx"] = value



class ProgressBar(object):
    """
    Print pretty progress bar
    """       
    def __init__(self, njobs, start, message):
        self.njobs = njobs
        self.start = time.time()
        self.message = message
        self.finished = 0
        
    @property
    def progress(self):
        return 100 * (self.finished / float(self.njobs))

    @property
    def elapsed(self):
        return datetime.timedelta(seconds=int(time.time() - self.start))
 
    def update(self):
        # build the bar
        hashes = '#' * int(self.progress / 5.)
        nohash = ' ' * int(20 - len(hashes))

        # print to stderr
        print("\r[{}] {:>3}% {} | {:<12} ".format(*[
            hashes + nohash,
            int(self.progress),
            self.elapsed,
            self.message,
        ]), end="")
        sys.stdout.flush()



# used by resolve_ambigs
GETCONS = np.array([
    [82, 71, 65],
    [75, 71, 84],
    [83, 71, 67],
    [89, 84, 67],
    [87, 84, 65],
    [77, 67, 65]], dtype=np.uint8,
)

# the three indexed resolutions of each quartet
TESTS = np.array([
    [0, 1, 2, 3], 
    [0, 2, 1, 3], 
    [0, 3, 1, 2]], dtype=np.uint8,
)
