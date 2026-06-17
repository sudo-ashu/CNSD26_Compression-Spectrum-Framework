import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import ECoG_compression_main as ecog_main


awake_df = pd.read_csv("AwakeCSV.csv")
anes_df = pd.read_csv("AnesthetizedCSV.csv")
sleep_df = pd.read_csv("SleepCSV.csv")

awake_bw = awake_df["bandwidth"]
anes_bw = anes_df["bandwidth"]
sleep_bw = sleep_df["bandwidth"]

# print(awake_bw.shape)
# print(anes_bw.shape)
# print(sleep_bw.shape)

awake_anes = np.concatenate([awake_bw, anes_bw])
awake_sleep = np.concatenate([awake_bw, sleep_bw])
sleep_anes = np.concatenate([sleep_bw, anes_bw])

AA_bins = np.linspace( awake_anes.min(), awake_anes.max(), 12)
AS_bins = np.linspace( awake_sleep.min(), awake_sleep.max(), 12)
SA_bins = np.linspace( sleep_anes.min(), sleep_anes.max(), 12)

plt.figure(figsize=(8,6))
plt.hist(awake_bw, bins=AA_bins, alpha=0.5, label="Awake")
plt.hist(anes_bw, bins=AA_bins, alpha=0.6,label="Anesthetized")
plt.xlabel("Bandwidth")
plt.ylabel("No. of Recordings")
plt.title("Bandwidth Distribution")
plt.legend()

plt.figure(figsize=(8,6))
plt.hist(awake_bw, bins=AS_bins, alpha=0.5, label="Awake")
plt.hist(sleep_bw, bins=AS_bins, alpha=0.6,label="Sleep")
plt.xlabel("Bandwidth")
plt.ylabel("No. of Recordings")
plt.title("Bandwidth Distribution")
plt.legend()

plt.figure(figsize=(8,6))
plt.hist(awake_bw, bins=SA_bins, alpha=0.5, label="Sleep")
plt.hist(anes_bw, bins=SA_bins, alpha=0.6,label="Anesthetized")
plt.xlabel("Bandwidth")
plt.ylabel("No. of Recordings")
plt.title("Bandwidth Distribution")
plt.legend()

# Box-plots

awake_df["state"] = "Awake"
sleep_df["state"] = "Sleep"
anes_df["state"] = "Anesthetized"

features = [
    "ETC",
    "bandwidth",
    "max_scale",
    "fluctuation",
    # "mean_entropy"
]

combined = pd.concat([awake_df, sleep_df, anes_df], ignore_index=True)

for feature in features:

    plt.figure(figsize=(8,6))

    sns.boxplot(
        x="state",
        y=feature,
        data=combined
    )

    plt.xlabel("States")
    plt.ylabel(f"{feature}")
    plt.title(f"{feature} across States")

plt.show()

# 2. between awake and sleep

# 3. between sleep and anethetic