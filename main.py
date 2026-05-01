import numpy as np
import compression_spec_scale_info as comp_spec
import plottings as plts
import shannon_eegfilt_partition_findPair_substitute as bts
import matplotlib.pyplot as plt

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
num_bins = 4
print(len(comp_spec.compression_spectrum_scale_info_NEW(X_filt, num_bins)))
comp_ratio, N, scale, scale_comp_cell, Ent = comp_spec.compression_spectrum_scale_info_NEW(X_filt, num_bins)

print("ETC value:", N)
print("Spectrum bandwidth:", np.count_nonzero(comp_ratio))
print("Max scale:", max(scale))
print("Mean fluctuation:", np.mean(np.abs(np.diff(scale))))

# PLOTS
plts.plot_all(comp_ratio, scale, Ent, N, num_bins)
plts.plot_frequency_spectrum(X_filt, fs)
plts.plot_compression_vs_scale(comp_ratio)
plts.plot_entropy_vs_scale(Ent)

# return comp_ratio, scale, Ent

# Now running it for all the 100 reading...
import os
import pandas as pd

def run_all_channels(folder_path, num_channels=100):

    results = []

    for i in range(1, num_channels + 1):
        filename = f"F{str(i).zfill(3)}.txt"
        filepath = os.path.join(folder_path, filename)

        # if not os.path.exists(filepath):
        #     print(f"Missing: {filename}")
        #     continue

        print(f"Processing {filename}")

        X = np.loadtxt(filepath)

        # Filter (FIR) ---
        fs = 173.61
        X_filt = bts.eegfilt_equivalent(X, fs, 0.53, 40)

        # Trimming edges
        X_filt = X_filt[50:-50]

        comp_ratio, N, scale, scale_comp_cell, Ent = comp_spec.compression_spectrum_scale_info_NEW(X_filt, num_bins)

        # Extract iteration-wise scale
        scale_iter = scale[num_bins:num_bins + N]

        # Features
        max_scale = max(scale)
        bandwidth = np.count_nonzero(comp_ratio)
        fluctuation = np.mean(np.abs(np.diff(scale_iter))) if len(scale_iter) > 1 else 0

        results.append({
            "channel": i,
            "file": filename,
            "ETC": N,
            "bandwidth": bandwidth,
            "max_scale": max_scale,
            "fluctuation": fluctuation
        })

    return pd.DataFrame(results)

folder = "F"
df = run_all_channels(folder)

print(df.head())

df.to_csv("EEG_100ch_results.csv", index=False)
print("Saved EEG_100ch_results.csv")

# ETC across channels
plt.plot(df["channel"], df["ETC"], marker='o')
plt.xlabel("Channel")
plt.ylabel("ETC")
plt.title("ETC across 100 EEG Channels")
plt.grid()
plt.show()

# Max Scale
plt.plot(df["channel"], df["max_scale"], marker='o')
plt.xlabel("Channel")
plt.ylabel("Max Scale")
plt.title("Max Scale across Channels")
plt.grid()
plt.show()

# Fluctuations
plt.plot(df["channel"], df["fluctuation"], marker='o')
plt.xlabel("Channel")
plt.ylabel("Fluctuation")
plt.title("Scale Fluctuation across Channels")
plt.grid()
plt.show()


def plot_mean_trial(folder_path):

    all_scales = []

    for i in range(1, 101):

        filename = f"F{str(i).zfill(3)}.txt"
        filepath = os.path.join(folder_path, filename)

        if not os.path.exists(filepath):
            continue

        signal = np.loadtxt(filepath)

        _, N, scale, _, _ = comp_spec.compression_spectrum_scale_info_NEW(signal, 4)

        scale_iter = scale[4:4 + N]
        all_scales.append(scale_iter)

    # Pad sequences
    max_len = max(len(s) for s in all_scales)
    padded = np.full((len(all_scales), max_len), np.nan)

    for i, s in enumerate(all_scales):
        padded[i, :len(s)] = s

    mean_scale = np.nanmean(padded, axis=0)
    std_scale = np.nanstd(padded, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_scale, label="Mean Scale")
    plt.fill_between(range(len(mean_scale)),
                     mean_scale - std_scale,
                     mean_scale + std_scale,
                     alpha=0.3)

    plt.xlabel("Iteration")
    plt.ylabel("Scale")
    plt.title("Mean Scale  TD across 100 trials)")
    plt.legend()
    plt.grid()
    plt.show()

folder = "F"
plot_mean_trial(folder)