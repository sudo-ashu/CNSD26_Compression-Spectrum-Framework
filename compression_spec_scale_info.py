import numpy as np
import shannon_eegfilt_partition_findPair_substitute as bts

# FULL COMPRESSION SPECTRUM 
def compression_spectrum_scale_info_NEW(signal, num_bins):

    sym_seq = bts.partition(signal, num_bins)
    Hnew = bts.shannon_entropy(sym_seq)

    thresh = len(sym_seq)
    comp_log = np.zeros(thresh)

    # symbol → creation index mapping
    sym2idx = {}
    scale = []
    scale_comp = []

    # initialize original symbols
    for s in range(1, num_bins + 1):
        sym2idx[s] = len(scale)
        scale.append(1)

    sym_current = sym_seq.copy()
    N = 0
    length = len(sym_current)


    while Hnew > 1e-6 and len(sym_current) > 1 and N < thresh:

        pair = bts.find_pair2(sym_current)
        sym_new, rep_sym = bts.substitute(sym_current, pair)

        new_len = len(sym_new)
        # if length == new_len + 1:
        #     break

        frac = length / new_len

        idx1 = sym2idx[int(pair[0])]
        idx2 = sym2idx[int(pair[1])]

        scale_val = scale[idx1] + scale[idx2]

        # register new symbol
        sym2idx[rep_sym] = len(scale)
        scale.append(scale_val)
        scale_comp.append([scale[idx1], scale[idx2]])

        # comp_log[scale_val - 1] += np.log2(frac)
        if scale_val - 1 < len(comp_log):   # now it won't overflow
            comp_log[scale_val - 1] += np.log2(frac)

        # update
        sym_current = sym_new
        length = new_len
        Hnew = bts.shannon_entropy(sym_current)
        N += 1

    comp_ratio = comp_log[:max(scale)]

    max_scale = max(scale)
    scale_comp_cell = [[] for _ in range(max_scale)]

    # only constructed symbols (after initial bins)
    for i, comp in enumerate(scale_comp):
        sc = scale[num_bins + i]
        scale_comp_cell[sc - 1].append(comp)

    Ent = np.zeros(max_scale)
    for i in range(max_scale):
        comps = scale_comp_cell[i]
        if len(comps) == 0:
            Ent[i] = 0
        else:
            comps = np.array(comps)
            if comps.ndim == 1:
                comps = comps.reshape(-1, 2)
            comps_sorted = np.sort(comps, axis=1)

            # unique rows
            ic = np.unique(comps_sorted, axis=0, return_inverse=True)[1]
            Ent[i] = bts.shannon_entropy(ic + 1)

    return comp_ratio, N, scale, scale_comp_cell, Ent