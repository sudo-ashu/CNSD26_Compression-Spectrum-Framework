from scipy import stats
from scipy.stats import shapiro
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


awake_df = pd.read_csv("AwakeCSV.csv")
anes_df = pd.read_csv("AnesthetizedCSV.csv")
sleep_df = pd.read_csv("SleepCSV.csv")

features = [
    "ETC",
    "bandwidth",
    "max_scale",
    "fluctuation",
    "mean_entropy"
]

datasets = {
    "Awake": awake_df,
    "Sleep": sleep_df,
    "Anesthetized": anes_df
}

for feature in features:
    print(f"\n===== {feature} =====")

    for state, df in datasets.items():
        stat, p = shapiro(df[feature])
        print(f"{state:15s}  p = {p:.4f}")


# visual test: Q-Q plot
stats.probplot(
    sleep_df["ETC"],
    dist="norm",
    plot=plt
)

plt.title("Sleep - Bandwidth Q-Q Plot")
plt.show()