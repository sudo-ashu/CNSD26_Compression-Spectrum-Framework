import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import compression_spec_scale_info as comp_spec
import ECoG_plottings as comp_spec_plot


def compute_session_features_full(df):
    session_results = []

    for session in df["session"].unique():
        df_sess = df[df["session"] == session]

        session_results.append({
            "session": session,

            "ETC_mean": df_sess["ETC"].mean(),
            "ETC_std": df_sess["ETC"].std(),

            "bandwidth_mean": df_sess["bandwidth"].mean(),
            "bandwidth_std": df_sess["bandwidth"].std(),

            "max_scale_mean": df_sess["max_scale"].mean(),
            "max_scale_std": df_sess["max_scale"].std(),

            "fluctuation_mean": df_sess["fluctuation"].mean(),
            "fluctuation_std": df_sess["fluctuation"].std(),
        })
    return pd.DataFrame(session_results)



def run_ecog_bins(base_folder):
    results = []

    for session in sorted(os.listdir(base_folder)):
        session_path = os.path.join(base_folder, session)

        if not os.path.isdir(session_path):
            continue

        print(f"Processing {session}")

        for file in sorted(os.listdir(session_path)):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(session_path, file)

            # --- Load bin ---
            df = pd.read_csv(file_path)
            data = df.values   # shape: (time, channels)

            num_channels = data.shape[1]
            num_symbols = 4

            for ch in range(num_channels):

                signal = data[:, ch]

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
                    "session": session,
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

                print("ETC value:", N)
                print("Spectrum bandwidth:", np.count_nonzero(comp_ratio))
                print("Max scale:", max(scale))
                print("Mean fluctuation:", np.mean(np.abs(np.diff(scale))))

    return pd.DataFrame(results)

df_results = run_ecog_bins("selected_bins_csv")

df_results.to_csv("ECoG_compression_features.csv", index=False)
print("Saved ECoG_compression_features.csv")

compute_session_features_full(df_results)
comp_spec_plot.plot_ECoG_Session(df_results)