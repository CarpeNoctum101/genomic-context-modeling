import numpy as np

PAD = 0

nuc = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4
}

VOCAB_SIZE = len(nuc) + 1

def tokenize_sequence(seq):
    return np.array([nuc.get(base, PAD) for base in seq], dtype=np.int64)