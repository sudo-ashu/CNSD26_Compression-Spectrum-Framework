import numpy as np
import compression_spec_scale_info
import plottings
import shannon_eegfilt_partition_findPair_substitute as bts

file_path = "F056.txt"
X = np.loadtxt(file_path)

fs = 173.61
lowcut = 0.53
highcut = 40

# APPLY FILTER
X_filt = bts.eegfilt_equivalent(X, fs, lowcut, highcut)

# TRIM
X_filt = X_filt[50:-50]

# COMPRESSION SPECTRUM
print(len(compression_spec_scale_info.compression_spectrum_scale_info_NEW(X_filt, 4)))
comp_ratio, N, scale, scale_comp_cell, Ent = compression_spec_scale_info.compression_spectrum_scale_info_NEW(X_filt, 4)

print("ETC value:", N)
print("Spectrum bandwidth:", np.count_nonzero(comp_ratio))
print("Max scale:", max(scale))
print("Mean fluctuation:", np.mean(np.abs(np.diff(scale))))

# PLOTS
plottings.plot_all(comp_ratio, scale, Ent)
plottings.plot_frequency_spectrum(X_filt, fs)
plottings.plot_compression_vs_scale(comp_ratio)
plottings.plot_entropy_vs_scale(Ent)

# return comp_ratio, scale, Ent