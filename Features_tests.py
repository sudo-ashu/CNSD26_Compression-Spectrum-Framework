from scipy import stats
from scipy.stats import shapiro, f_oneway, kruskal
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


########### visual test: Q-Q plot ##########
stats.probplot(
    sleep_df["ETC"],
    dist="norm",
    plot=plt
)

plt.title("Sleep - Bandwidth Q-Q Plot")
plt.show()


############################################
# using z_score test to test for outliers

from scipy.stats import zscore

sleep_df["ETC_z"] = zscore(sleep_df["ETC"])
sleep_df["Entropy_z"] = zscore(sleep_df["mean_entropy"])

print(sleep_df[np.abs(sleep_df["ETC_z"]) > 3])
print(sleep_df[np.abs(sleep_df["Entropy_z"]) > 3])

sleep_clean = sleep_df[np.abs(sleep_df["ETC_z"]) < 3]
stat, p = shapiro(sleep_clean["ETC"])
print(p)

###########################################
# ANOVA test for the anova-group

for feature in features[1:4]:
    F, p = f_oneway(
        awake_df[feature],
        sleep_df[feature],
        anes_df[feature]
    )
    print("F-statistic =", F)
    print("p-value =", p)

###########################################
# Kruksal-Wallis test for the ETC

H, p = kruskal(
    awake_df["ETC"],
    sleep_df["ETC"],
    anes_df["ETC"]
)

print("H-statistic =", H)
print("p-value =", p)