"""A series of utilities for dealing with sequences.
"""
import numpy as np

"""
    Note: DNA sequences are always stored from 5' end to 3' end by convention.

    5'-AGGT-3'
    ||||
    3'-TCCA-5'

    sequence: 5'-AGGT-3'
    complement: 3'-TCCA-5'
    reverse complement: 5'-ACCT-3'
"""
# DNA bases (A, T, C, and G), where index maps to a base (0-A, 1-T, 2-C, 3-G)
bases = np.array(list("ATCG"))
# Generates a random sequence of length 'n'.
randseq = lambda n: "".join(np.random.choice(bases, n))
# Map of each base to its Watson-Crick complement.
complement = dict(zip("ATCG", "TAGC"))
# Reverse-complement (given a sequence s, returns the reverse-complement)
revcomp = lambda s: "".join(reversed([complement[b] for b in s]))


def seq_hdist(s1, s2):
    """Hamming distance [1] between two sequences.

    [1] - https://en.wikipedia.org/wiki/Hamming_distance

    Parameters
    ----------
    s1 : str
        The first DNA sequence.
    s2 : str
        The second DNA sequence.

    Returns
    -------
    np.float
        The hamming-distance between s1 and s2.
    """
    return np.mean(np.array(list(s1)) != np.array(list(s2)))

def mutate(seq, mut_rate = 0.5):
    """Randomly mutate a sequence with substitutions at a given mutation rate.
    No insertions or deletions.

    Parameters
    ----------
    seq : str
        The sequence to mutate.
    mut_rate : float, optional
        The probability of a substitution mutation occurring at each position, by default 0.5

    Returns
    -------
    str
        A mutated version of the original sequence.
    """
    seq_list = list(seq)
    for i,b in enumerate(seq_list):
        if np.random.random() < mut_rate:
            seq_list[i] = np.random.choice([base for base in bases if base != b])
    return "".join(seq_list)

def onehots_to_seqs(onehots):
    """Converts one-hot sequences (N x L x 4) to strings (N x L)
    Where N is the number of sequences and L is the length of each sequence.

    Parameters
    ----------
    onehots : np.array
        3D numpy array of either one-hot or soft-max encoded DNA sequences.

    Returns
    -------
    np.array
        Array of sequence strings.
    """

    return np.array([
        "".join(seq) for seq in bases[onehots.argmax(-1)]
    ])

def seqs_to_onehots(seqs):
    """Convert strings (N x L) to one-hot sequences (N x L x 4)
    Where N is the number of sequences and L is the length of each sequence.

    Parameters
    ----------
    seqs : Iterable
        Iterable of N number of DNA sequences of length L.

    Returns
    -------
    np.array
        The one-hot encoded sequences.
    """
    seq_array = np.array(map(list, seqs))
    return np.array([(seq_array == b).T for b in bases]).T.astype(int)


def random_mutant_pairs(n, d):
    """Generates 'n' pairs of random sequences of length 'd' where the hamming distances
    between the pairs are drawn from a uniform distribution.

    All sequence-pair distances are equally randomly.

    Parameters
    ----------
    n : int
        Number of pairs to generate.
    d : int
        The length of each sequence.

    Returns
    -------
    Tuple(np.array, np.array)
        Tuple containing (the array of sequence pairs, hamming-distance per pair)
    """
    targets = [ randseq(d) for _ in range(n) ]
    mut_rates = np.random.uniform(0, 1, size=n)

    pairs = np.array(
        [ [ target, mutate(target, rate) ]
          for target, rate in zip(targets, mut_rates)
        ]
    )
    seq_hdists = np.array(
        [ seq_hdist(s1, s2) for s1, s2 in pairs ]
    )

    return pairs, seq_hdists
