import numpy as np

bases = np.array(list("ATCG"))
randseq = lambda n: "".join(np.random.choice(bases, n))
complement = dict(zip("ATCG", "TAGC"))
revcomp = lambda s: "".join(reversed([complement[b] for b in s]))

def seq_hdist(s1, s2):
    return np.mean(np.array(list(s1)) != np.array(list(s2)))

def mutate(seq, mut_rate = 0.5):
    seq_list = list(seq)
    for i,b in enumerate(seq_list):
        if np.random.random() < mut_rate:
            seq_list[i] = np.random.choice([base for base in bases if base != b])
    return "".join(seq_list)

def onehots_to_seqs(onehots):
    """Convert one-hot sequences (N x L x 4) to strings (N x L)"""
    return np.array([
        "".join(seq) for seq in bases[onehots.argmax(-1)]
    ])

def seqs_to_onehots(seqs):
    """Convert strings (N x L) to one-hot sequences (N x L x 4)"""
    seq_array = np.array(map(list, seqs))
    return np.array([(seq_array == b).T for b in bases]).T.astype(int)


def random_mutant_pairs(n, d):
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
