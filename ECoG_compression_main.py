import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import compression_spec_scale_info as comp_spec
import ECoG_plottings as comp_spec_plot

def aggregate_state_unbiased(df):
    if df.empty:
        raise ValueError("Cannot aggregate: The input DataFrame is empty. Check your folder structure!")

    # Step 1: Mean per recording first (equalizes single-channel vs WB weight)
    df_rec = df.groupby(["state", "region", "recording"], as_index=False).agg({
        "ETC": "mean",
        "bandwidth": "mean",
        "max_scale": "mean",
        "fluctuation": "mean",
        "mean_entropy": "mean"
    })

    # Step 2: Compute mean & std across recordings for each state/region pair
    df_state = df_rec.groupby(["state", "region"]).agg({
        "ETC": ["mean", "std"],
        "bandwidth": ["mean", "std"],
        "max_scale": ["mean", "std"],
        "fluctuation": ["mean", "std"],
        "mean_entropy": ["mean", "std"]
    })

    df_state.columns = ['_'.join(col) for col in df_state.columns]
    return df_state.reset_index()


def run_ecog_bins(base_folder, num_symbols=4):
    results = []

    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"Base folder '{base_folder}' does not exist! Check your path.")

    # Level 1: State (Awake, Sleep, Anaesthetized)
    for state in sorted(os.listdir(base_folder)):
        state_path = os.path.join(base_folder, state)
        if not os.path.isdir(state_path) or state.startswith('.'):
            continue

        print(f"\n================ Processing State: {state} ================")

        # Level 2: Brain Region (HV, MP, LP, ..., WB)
        for region in sorted(os.listdir(state_path)):
            region_path = os.path.join(state_path, region)
            if not os.path.isdir(region_path) or region.startswith('.'):
                continue

            print(f"  --> Region: {region}")

            # Level 3: Recording (Recording_01 ... Recording_30)
            for recording in sorted(os.listdir(region_path)):
                recording_path = os.path.join(region_path, recording)
                if not os.path.isdir(recording_path) or recording.startswith('.'):
                    continue
                
                # Level 4
                for file in sorted(os.listdir(recording_path)):
                    if not file.endswith(".csv"):
                        continue

                    file_path = os.path.join(recording_path, file)

                    df = pd.read_csv(file_path, header=None) 
                    data = df.values  # (channels, 400)

                    if data.shape[0] > data.shape[1]: 
                        data = data.T  # Transpose if rows happen to be time points

                    # Dropping the index column if the layout is exactly 2 rows
                    if data.shape[0] == 2:
                        data = data[1:, :]  
                    # -----------------------

                    num_channels, num_timepoints = data.shape

                    for ch in range(num_channels):
                        signal = data[ch, :]

                        # Compression spectrum pipeline
                        comp_ratio, N, scale, scale_comp_cell, Ent = comp_spec.compression_spectrum_scale_info_NEW(
                            signal, num_symbols
                        )

                        scale_iter = scale[num_symbols : num_symbols + N]

                        # Feature extraction
                        max_scale = max(scale) if len(scale) > 0 else 0
                        bandwidth = np.count_nonzero(comp_ratio)
                        fluctuation = np.mean(np.abs(np.diff(scale_iter))) if len(scale_iter) > 1 else 0
                        mean_entropy = np.mean(Ent)

                        results.append({
                            "state": state,
                            "region": region,
                            "recording": recording,
                            "bin": file,
                            "channel": ch + 1,

                            "ETC": N,
                            "bandwidth": bandwidth,
                            "max_scale": max_scale,
                            "fluctuation": fluctuation,
                            "mean_entropy": mean_entropy,

                            "comp_ratio": comp_ratio.tolist() if hasattr(comp_ratio, "tolist") else comp_ratio,
                            "scale": scale.tolist() if hasattr(scale, "tolist") else scale, # type: ignore
                            "Ent": Ent.tolist() if hasattr(Ent, "tolist") else Ent
                        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"\n WARNING: No CSV files were processed in '{base_folder}'. Please verify your directory structure matches:")
        print(f"   {base_folder}/ <State>/ <Region>/ <Recording>/ <bin.csv>")
        
    return df


# --- Execution ---
df_results = run_ecog_bins("States")

if not df_results.empty:
    # Save full granular results
    df_results.to_csv("ECoG_Granular_Results.csv", index=False)

    # Generate and save aggregated summary
    df_states = aggregate_state_unbiased(df_results)
    df_states.to_csv("Features_State_Region_Summary.csv", index=False)

    print("\nProcessing complete!")
    print(f"Granular Results Shape: {df_results.shape}")
    print(f"Summary Results Shape:  {df_states.shape}")
