import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg

from scipy.stats import f_oneway, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#   Load the single granular dataset
df_all = pd.read_csv("ECoG_Granular_Results.csv")

# Clean up state names 
df_all['state'] = df_all['state'].str.replace('_Region_State', '', regex=False)
df_all['state'] = df_all['state'].replace({'Anaesthetized': 'Anesthetized'})

def plot_feature_by_region_smart(df, region, feature, ylabel=None, save=False):
    print(f"\n========== {feature.upper()} in REGION: {region} ==========")
    
    # Filter and Aggregate Data
    df_region = df[df["region"] == region]
    
    if df_region.empty:
        print(f"Error: No data found for region '{region}'.")
        return None

    # Aggregate to recording-level 
    df_rec = df_region.groupby(["state", "recording"], as_index=False)[feature].mean()

    # Extract arrays for each state
    awake = df_rec[df_rec["state"] == "Awake"][feature].values
    sleep = df_rec[df_rec["state"] == "Sleep"][feature].values
    anes = df_rec[df_rec["state"] == "Anesthetized"][feature].values
    
    if len(awake) == 0 or len(sleep) == 0 or len(anes) == 0:
        print("Warning: Missing data for one or more states in this region.")
        return None

    labels = ["Awake", "Sleep", "Anesthetized"]

    # Levene's Test
    stat, p_levene = levene(awake, sleep, anes)
    print(f"Levene's Test p-value: {p_levene:.4e}")

    if p_levene >= 0.05:
        print("--> Variances are equal. Using Standard 1-way ANOVA & Tukey HSD.")
        
        # Standard ANOVA
        F, p = f_oneway(awake, sleep, anes)
        title_stats = f"Standard ANOVA (p = {p:.2e})"
        print(f"F = {F:.4f}, p = {p:.3e}")

        # Tukey HSD
        values = np.concatenate([awake, sleep, anes])
        groups = (["Awake"] * len(awake) + ["Sleep"] * len(sleep) + ["Anesthetized"] * len(anes))
        tukey = pairwise_tukeyhsd(values, groups)
        
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        posthoc_df = tukey_df[['group1', 'group2', 'p-adj']].copy()
        
    else:
        print("--> Variances are Unequal. Using Welch's ANOVA & Games-Howell.")
        
        # Welch's ANOVA
        welch_results = pg.welch_anova(dv=feature, between='state', data=df_rec)
        F = welch_results['F'].values[0]
        p = welch_results['p_unc'].values[0]
        title_stats = f"Welch's ANOVA (p = {p:.2e})"
        print(f"F = {F:.4f}, p = {p:.3e}")

        # Games-Howell
        gh_results = pg.pairwise_gameshowell(dv=feature, between='state', data=df_rec)
        
        # Standardize columns to match the plotting logic
        posthoc_df = gh_results[['A', 'B', 'pval']].rename(
            columns={'A': 'group1', 'B': 'group2', 'pval': 'p-adj'}
        )

    print("\nPost-Hoc Results:")
    print(posthoc_df)

    # ----------------------------------------------------
    # Plotting
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8,7))
    colors = ["#4C9FD1", "#F39C12", "#27AE60"]

    bp = ax.boxplot(
        [awake, sleep, anes],
        widths=0.55,
        patch_artist=True,
        showfliers=False
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)

    for whisker in bp["whiskers"]: whisker.set_linewidth(2)
    for cap in bp["caps"]: cap.set_linewidth(2)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.5)

    # Scatter Jitter Overlay
    rng = np.random.default_rng(42)
    for i, data in enumerate([awake, sleep, anes], start=1):
        x = rng.normal(i, 0.05, len(data))
        ax.scatter(
            x, data,
            s=45, color=colors[i-1], edgecolor='black',
            linewidth=0.5, alpha=0.75, zorder=3
        )

    # Significance Brackets
    def stars(p_val):
        if p_val < 0.001: return "***"
        elif p_val < 0.01: return "**"
        elif p_val < 0.05: return "*"
        else: return "ns"

    comparisons = {
        ("Awake", "Sleep"): (1, 2),
        ("Awake", "Anesthetized"): (1, 3),
        ("Sleep", "Anesthetized"): (2, 3)
    }

    # Calculate y-limits dynamically
    values_all = np.concatenate([awake, sleep, anes])
    ymax = values_all.max()
    ymin = values_all.min()
    spacing = (ymax - ymin) * 0.10
    height = ymax + spacing

    for _, row in posthoc_df.iterrows():
        pair = (row["group1"], row["group2"])
        if pair not in comparisons:
            pair = (row["group2"], row["group1"])
            
        x1, x2 = comparisons[pair]
        label = stars(row["p-adj"])

        ax.plot(
            [x1, x1, x2, x2],
            [height, height + spacing/4, height + spacing/4, height],
            lw=2, color='black'
        )

        ax.text(
            (x1 + x2) / 2,
            height + spacing/3,
            label,
            ha='center', fontsize=18, fontweight='bold'
        )
        height += spacing

    # Formatting
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=16)

    if ylabel is None:
        ylabel = feature.replace("_", " ").title()

    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(f"{ylabel} in {region}\n{title_stats}", fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.set_ylim(ymin - spacing, height + spacing)
    plt.tight_layout()

    if save:
        filename = f"{feature}_{region}.png"
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        print(f"Saved plot as {filename}")
        
    plt.show()

all_regions = df_all['region'].unique()
for r in all_regions:
    plot_feature_by_region_smart(df=df_all, region=r, feature="bandwidth", ylabel="Spectral Bandwidth", save=True)

all_regions = df_all['region'].unique()
for r in all_regions:
    plot_feature_by_region_smart(df=df_all, region=r, feature="max_scale", ylabel="Maximum scale in the spectrum", save=True)
