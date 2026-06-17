import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import compression_spec_scale_info as comp_spec
import ECoG_plottings as comp_spec_plot


def aggregate_state(df):

    df_state = df.groupby("state").agg({
        "ETC": ["mean", "std"],
        "bandwidth": ["mean", "std"],
        "max_scale": ["mean", "std"],
        "fluctuation": ["mean", "std"],
        "mean_entropy": ["mean", "std"]
    })

    # Flatten column names
    df_state.columns = ['_'.join(col) for col in df_state.columns]
    df_state = df_state.reset_index()

    return df_state



def run_ecog_bins(base_folder):
    results = []

# if the state folder is also there under some folder then this loop is mandatory

    # for state in sorted(os.listdir(base_folder)):
    #     state_path = os.path.join(base_folder, state)
    #     if not os.path.isdir(state_path):
    #         continue

    #     print(f"\nProcessing State: {state}")

    for recording in sorted(os.listdir(base_folder)):
        recording_path = os.path.join(base_folder, recording)
        if not os.path.isdir(recording_path):
            continue

        print(f"   Processing Recording: {recording}")

        for file in sorted(os.listdir(recording_path)):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(recording_path, file)

            # --- Load bin ---
            df = pd.read_csv(file_path)
            data = df.values   # shape: (time, channels)

            num_channels = data.shape[0]
            num_symbols = 4

            for ch in range(num_channels):
                signal = data[ch, :]
                # signal = eegfilt_equivalent(signal, fs, 0.53, 40)  -------optional (The ECoG data is already filtered)

                # Running compression spectrum file
                comp_ratio, N, scale, scale_comp_cell, Ent = comp_spec.compression_spectrum_scale_info_NEW(signal, num_symbols)

                # Extracting iteration scale
                scale_iter = scale[num_symbols:num_symbols + N]

                # Features
                max_scale = max(scale) if len(scale) > 0 else 0
                bandwidth = np.count_nonzero(comp_ratio)
                fluctuation = np.mean(np.abs(np.diff(scale_iter))) if len(scale_iter) > 1 else 0
                mean_entropy = np.mean(Ent)

                results.append({

                    # "state": state,
                    "recording": recording,
                    "bin": file,
                    "channel": ch + 1,

                    "ETC": N,
                    "bandwidth": bandwidth,
                    "max_scale": max_scale,
                    "fluctuation": fluctuation,
                    "mean_entropy": mean_entropy,

                    "comp_ratio": comp_ratio.tolist(),
                    "scale": scale,
                    "Ent": Ent.tolist()

                })

    return pd.DataFrame(results)

df_results = run_ecog_bins("Sleep_state")

# df_results.to_csv("AnesthetizedCSV.csv", index=False)
# print("Saved ECoG_compression_features2.csv")

# df_states = aggregate_state(df_results)
# df_states.to_csv("Features_csv4.csv")
# comp_spec_plot.plot_ECoG_States(df_results)

df_recording = (df_results.groupby(["recording"]).agg({
        "ETC": "mean",
        "bandwidth": "mean",
        "max_scale": "mean",
        "fluctuation": "mean",
        "mean_entropy": "mean"
    }).reset_index()
)

df_recording.to_csv("SleepCSV.csv", index=False)
print(df_results.shape)
print(df_recording.shape)

#################################################################