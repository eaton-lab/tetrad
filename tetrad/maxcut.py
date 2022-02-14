#!/usr/bin/env python

"""Implementation of a quartet max-cut algorithm.

UNDER DEVELOPMENT, NOT YET IMPLEMENTED.

References
-----------
- Snir and Rao (2012).
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.html
- http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
"""

from typing import Set, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Vertex:
    name: str
    coordinate: Tuple[float, float, float]  # maybe array
    center_of_mass: Tuple[float, float, float] = (0, 0, 0)  # maybe array
    """Closest point to all neighbors, proportional to the edge weights of each neighbor"""

@dataclass
class Edge:
    vertices: Set[Vertex]
    weight: float

@dataclass
class Graph:
    vertices: Set[Vertex]
    edges: Set[Edge]


def random_three_vector():
    """Return a random uniform vector coordinate on surface of 3-d sphere.

    http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    """
    phi = np.random.uniform(0, np.pi * 2)
    cos_theta = np.random.uniform(-1, 1)

    theta = np.arccos(cos_theta)
    xco = np.sin(theta) * np.cos(phi)
    yco = np.sin(theta) * np.sin(phi)
    zco = np.cos(theta)
    return xco, yco, zco

def max_cut(quartets, niters: int=10):
    """

    """
    # get number of unique tips
    # ntips = len(...)

    # place vertices randomly on 3-d sphere
    vertices = [Vertex(i, random_three_vector()) for i in range(ntips)]

    # move every vertex towards its center_of_mass, repeated N times.
    # ...

    # draw hyperplane through origin of sphere, dividing vertex sets into A|B
    # ...


def get_quartet_graph():
    """

    Vertices are sample (names), edges are defined by input quartets.
    Each quartet adds 4 "good" edges, and 2 "bad" edges, the former
    get positive weight, the latter negative weights.

    """

def make_cut():
    """

    A cut is any bipartition that divides the vertex set (names)
    into two sets. An edge is "in the cut" if its two vertices are 
    on different sides of the bipartition. The weight of the cut
    is the sum of weights of the edges "in the cut".
    """

def cut_stats():
    """

    Unaffected quartets: vertices in quartet on same side of a cut.
    Affected quartets: 
        - satisfied: 
        - violated: 
        - deferred: 
    """

def max_cut_traversal():
    """
    
    Largest split at root, recursively travel down each descendant 
    assigning a cut to every node. Then travel back up the tree to 
    build the Nodes ...
    """

if __name__ == "__main__":

    import itertools

    NTIPS = 6
    TAXA = [str(i) for i in range(NTIPS)]
    QUARTS = list(itertools.combinations(TAXA, 4))
    print(QUARTS)

    # VERTS = [Vertex(str(i)) for i in range(20)]
    # EDGES = [Edge(vertices=set)]
