#!/usrbin/env python

"""

"""

import numpy as np
from numba import njit
from tetrad.src.utils import GETCONS


@njit
def jit_resolve_ambigs(tmpseq: np.ndarray, seed: int) -> np.ndarray:
    """Randomly resolve ambiguous bases in a uint8 seqarray. 

    This is applied to each boot replicate so that over reps the 
    random resolutions don't matter. Sites are randomly resolved, 
    so best for unlinked SNPs since otherwise linked SNPs are losing 
    their linkage information... though it's not like we're using
    it anyways.
    """
    np.random.seed(seed)

    # iter rows in GETCONS (np.uint([82, 75, 83, 89, 87, 77]))
    for aidx in range(6):
        ambig, res1, res2 = GETCONS[aidx]

        # get wher tmpseq == ambig and assign half to each resolution
        idx, idy = np.where(tmpseq == ambig)
        halfmask = np.random.binomial(n=1, p=0.5, size=idx.shape[0])
        for col in range(idx.shape[0]):
            if halfmask[col]:
                tmpseq[idx[col], idy[col]] = res1
            else:
                tmpseq[idx[col], idy[col]] = res2
    return tmpseq


if __name__ == "__main__":
    pass
