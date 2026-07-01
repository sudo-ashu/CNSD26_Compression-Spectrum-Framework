import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


awake_df = pd.read_csv("AwakeCSV.csv")
sleep_df = pd.read_csv("SleepCSV.csv")
anes_df = pd.read_csv("AnesthetizedCSV.csv")

def plot_feature(feature, ylabel=None, save=False):

    # ----------------------------------------------------
    # Extract feature
    # ----------------------------------------------------
    awake = awake_df[feature].values
    sleep = sleep_df[feature].values
    anes = anes_df[feature].values

    labels = ["Awake", "Sleep", "Anesthetized"]

    # ----------------------------------------------------
    # ANOVA
    # ----------------------------------------------------

    F, p = f_oneway(awake, sleep, anes)

    print(f"\n========== {feature.upper()} ==========")
    print(f"F = {F:.4f}")
    print(f"p = {p:.3e}")

    # ----------------------------------------------------
    # Tukey HSD
    # ----------------------------------------------------

    values = np.concatenate([awake, sleep, anes])

    groups = (
        ["Awake"] * len(awake)
        + ["Sleep"] * len(sleep)
        + ["Anesthetized"] * len(anes)
    )

    tukey = pairwise_tukeyhsd(values, groups)

    print(tukey)

    tukey_df = pd.DataFrame(
        tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8,7))

    colors = [
        "#4C9FD1",
        "#F39C12",
        "#27AE60"
    ]

    bp = ax.boxplot(
        [awake, sleep, anes],
        widths=0.55,
        patch_artist=True,
        showfliers=False
    )

    # Boxes

    for patch, color in zip(bp["boxes"], colors):

        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)

    # Whiskers

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(2)

    for cap in bp["caps"]:
        cap.set_linewidth(2)

    # Median

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.5)

    # ----------------------------------------------------
    # Scatter
    # ----------------------------------------------------

    rng = np.random.default_rng(42)

    for i, data in enumerate([awake, sleep, anes], start=1):

        x = rng.normal(i, 0.05, len(data))

        ax.scatter(
            x,
            data,
            s=45,
            color=colors[i-1],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.75,
            zorder=3
        )

    # ----------------------------------------------------
    # Significance function
    # ----------------------------------------------------

    def stars(p):

        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    comparisons = {
        ("Awake","Sleep"):(1,2),
        ("Awake","Anesthetized"):(1,3),
        ("Sleep","Anesthetized"):(2,3)
    }

    ymax = values.max()
    ymin = values.min()

    spacing = (ymax-ymin)*0.10

    height = ymax + spacing

    for _, row in tukey_df.iterrows():

        if (row["group1"], row["group2"]) in comparisons:

            x1,x2 = comparisons[(row["group1"],row["group2"])]

        else:

            x1,x2 = comparisons[(row["group2"],row["group1"])]

        label = stars(row["p-adj"])

        ax.plot(
            [x1,x1,x2,x2],
            [height,height+spacing/4,height+spacing/4,height],
            lw=2,
            color='black'
        )

        ax.text(
            (x1+x2)/2,
            height+spacing/3,
            label,
            ha='center',
            fontsize=18,
            fontweight='bold'
        )

        height += spacing

    # ----------------------------------------------------
    # Formatting
    # ----------------------------------------------------
    ax.set_xticks([1,2,3])
    ax.set_xticklabels( labels, fontsize=16)

    if ylabel is None:
        ylabel = feature.replace("_"," ").title()

    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(f"{ylabel}\nOne-way ANOVA (p = {p:.2e})", fontsize=18, fontweight='bold')
    ax.tick_params(axis='y',labelsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.set_ylim(ymin-spacing, height+spacing)
    plt.tight_layout()

    if save:
        plt.savefig( f"{feature}.png", dpi=600, bbox_inches="tight")
    plt.show()
    return tukey_df

plot_feature("max_scale", "Maximum scale in the spectrum for Conscious State", save=True)