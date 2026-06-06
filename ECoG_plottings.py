import numpy as np
import pandas as pd
import compression_spec_scale_info as comp_spec
import matplotlib.pyplot as plt
import os

def plot_ECoG_Session(df, num_symbols=4):

    sessions = df["session"].unique()

    for session in sessions:

        df_sess = df[df["session"] == session]

        all_comp = []
        all_scales = []
        all_entropy = []

        # --- collect ---
        for _, row in df_sess.iterrows():

            comp = np.array(row["comp_ratio"])
            scale = row["scale"]
            Ent = np.array(row["Ent"])
            N = row["ETC"]

            all_comp.append(comp)

            # iteration-scale
            scale_iter = scale[num_symbols:num_symbols + N]
            all_scales.append(scale_iter)

            all_entropy.append(Ent)

        # --- pad comp_ratio ---
        max_len_c = max(len(c) for c in all_comp)
        padded_c = np.zeros((len(all_comp), max_len_c))

        for i, c in enumerate(all_comp):
            padded_c[i, :len(c)] = c

        mean_comp = np.mean(padded_c, axis=0)

        # --- pad scale ---
        max_len_s = max(len(s) for s in all_scales)
        padded_s = np.full((len(all_scales), max_len_s), np.nan)

        for i, s in enumerate(all_scales):
            padded_s[i, :len(s)] = s

        mean_scale = np.nanmean(padded_s, axis=0)

        # --- pad entropy ---
        max_len_e = max(len(e) for e in all_entropy)
        padded_e = np.full((len(all_entropy), max_len_e), np.nan)

        for i, e in enumerate(all_entropy):
            padded_e[i, :len(e)] = e

        mean_entropy = np.nanmean(padded_e, axis=0)

        # --- PLOTTING ---
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # 1️⃣ Compression spectrum (STEM)
        scales = np.arange(1, len(mean_comp) + 1)
        valid = mean_comp > 0

        scale = scales[valid]
        log_comp = mean_comp[valid]

        markerline, stemlines, baseline = axs[0].stem(scale, log_comp)
        axs[0].set_title(f"{session} - Compression Spectrum")
        axs[0].set_xlabel("Scale")
        axs[0].set_ylabel("log2(Compression Ratio)")
        axs[0].grid()

        # 2️⃣ Scale vs Iteration
        axs[1].stairs(mean_scale, np.arange(len(mean_scale) + 1))
        axs[1].set_title(f"{session} - Scale vs Iteration")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Scale")
        axs[1].grid()

        # 3️⃣ Entropy vs Scale
        scales_e = np.arange(1, len(mean_entropy) + 1)
        axs[2].plot(scales_e, mean_entropy, marker='o')
        axs[2].set_title(f"{session} - Entropy vs Scale")
        axs[2].set_xlabel("Scale")
        axs[2].set_ylabel("Entropy")
        axs[2].grid()

        plt.tight_layout()
        plt.show()