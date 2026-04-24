import numpy as np
from scipy.signal import firwin, filtfilt

# Shannon Entropy
def shannon_entropy(seq):
    values, counts = np.unique(seq, return_counts=True)
    prob = counts / len(seq)
    return -np.sum(prob * np.log2(prob + 1e-12))


def eegfilt_equivalent(signal, fs, lowcut, highcut):
    nyq = fs / 2
    taps = firwin(401, [lowcut/nyq, highcut/nyq], pass_zero=False)
    return filtfilt(taps, [1.0], signal)


# Partition
def partition(signal, num_bins):
    x = signal - np.min(signal)
    delta = (np.max(signal) - np.min(signal) + 1e-6) / num_bins
    sym = np.floor(x / delta).astype(int) + 1
    return sym


# FindPair2 
def find_pair2(sym_seq):
    M = int(np.max(sym_seq))
    count_array = np.zeros((M+1, M+1))

    L = len(sym_seq)
    i = 0

    while i < L - 1:
        a = int(sym_seq[i])
        b = int(sym_seq[i+1])
        count_array[a, b] += 1

        if a == b and i < L - 2 and sym_seq[i+2] == a:
            i += 1

        i += 1

    m = np.max(count_array)
    pos = np.argwhere(count_array == m)

    row = pos[:, 0]
    col = pos[:, 1]

    min_row = np.min(row)
    min_col = np.min(col)

    # MATLAB-style tie-breaking
    for r, c in zip(row, col):
        if r == min_row and c == min_col:
            return np.array([r, c])

    return pos[0]


# Substitute
def substitute(sym_seq, pair):
    new_seq = []
    rep_sym = int(np.max(sym_seq)) + 1

    i = 0
    L = len(sym_seq)

    while i < L - 1:
        if sym_seq[i] == pair[0] and sym_seq[i+1] == pair[1]:
            new_seq.append(rep_sym)
            i += 2
        else:
            new_seq.append(sym_seq[i])
            i += 1

    if i == L - 1:
        new_seq.append(sym_seq[-1])

    return np.array(new_seq), rep_sym