#!/usr/bin/env python

"""Split combinatorial quartet jobs into generator instructions.

"""


from itertools import combinations, islice
from math import comb
from numpy.random import default_rng


def get_chunks_info(nsamples: int, max_chunk_size: int) -> list[tuple[int, int]]:
    """Splits the combinations generator into chunks.

    Parameters
    ----------
    nsamples: int
        The size of the input set.
    max_chunk_size: int
        Max size of number of quartets in a chunk.

    Returns
    -------
        List of tuples containing the range of indices for each chunk.
    """
    total_combinations = comb(nsamples, 4)  # Total number of combinations
    chunks = []
    start = 0

    while start < total_combinations:
        end = min(start + max_chunk_size, total_combinations)
        chunks.append((start, end))
        start = end

    return chunks


def get_combinations_from_chunk(nsamples: int, start: int, end: int):
    """Extracts a specific chunk of combinations by index range.

    Parameters
    ----------
    nsamples: int
        The size of the input set.
    start: int
        Start index of the chunk.
    end: int
        End index of the chunk.

    Returns
    -------
        Generator of combinations for the specified range.
    """
    return islice(combinations(range(nsamples), 4), start, end)


def sample_combinations(nsamples: int, num_samples: int, rng: int):
    """Efficiently sample a random subset of combinations without generating all combinations.

    nsamples: The total number of samples to draw from.
    num_samples: Number of random samples to return.
    """
    rng = default_rng(rng)

    # Create a combinations generator
    comb_gen = combinations(range(nsamples), 4)

    # Reservoir sampling: Maintain a reservoir of 'num_samples' random combinations
    reservoir = []
    for i, qrt in enumerate(comb_gen):
        if i < num_samples:
            reservoir.append(qrt)
        else:
            j = rng.integers(0, i)
            if j < num_samples:
                reservoir[j] = qrt

    return reservoir


def iter_chunks_full(nsamples: int, max_size: int):
    """Generator of chunks of quartet samples over the total number.
    """
    chunk_ranges = get_chunks_info(nsamples, max_size)
    for i, (start, end) in enumerate(chunk_ranges):
        yield get_combinations_from_chunk(nsamples, start, end)


######################################################################
######################################################################


def _index_to_combination(index: int, n: int):
    """Convert a rank/index to a combination using combinatorial math
    for quartet samples.
    """
    combination = []
    for i in range(n):
        if len(combination) == 4:
            break
        if comb(n - i - 1, 4 - len(combination) - 1) > index:
            combination.append(i)
        else:
            index -= comb(n - i - 1, 4 - len(combination) - 1)
    return tuple(combination)


def random_combination_sample_via_index(nsamples: int, size: int, rng: int):
    """..."""
    total_combinations = comb(nsamples, 4)
    rng = default_rng(rng)
    sampled_indices = rng.choice(total_combinations, size=size, replace=False)
    return [_index_to_combination(idx, nsamples) for idx in sampled_indices]


def iter_chunks_random(nsamples: int, size: int, max_size: int, rng: int):
    """Generator of chunks of random quartet samples over the sampled number."""
    qrts = random_combination_sample_via_index(nsamples, size, rng)
    for i in range(0, len(qrts), max_size):
        yield qrts[i: i + max_size]




if __name__ == "__main__":

    # iterate over ordered chunks of 1000 qrts from total quartets
    qiter = iter_chunks_full(100, 100)
    qrts = next(qiter)
    for i in qrts:
        print(i)

    # iterate over random chunks of 100 qrts from 5000 sampled of <<< total 
    qiter = iter_chunks_random(200, 5000, 100, 123)
    qrts = next(qiter)
    for i in qrts:
        print(i)
