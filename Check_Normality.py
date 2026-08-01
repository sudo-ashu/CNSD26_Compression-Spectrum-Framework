import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot

def check_normality_assumptions(df, region, feature):

    print(f"\nNORMALITY CHECK: {feature.upper()} in REGION: {region}")
    
    # 1. Filter and Aggregating the data
    df_region = df[df["region"] == region]
    
    if df_region.empty:
        print(f"Error: No data found for region '{region}'.")
        return

    # Aggregate to recording-level
    df_rec = df_region.groupby(["state", "recording"], as_index=False)[feature].mean()

    # Extract arrays (using str.contains handles slight naming variations perfectly)
    awake = df_rec[df_rec["state"].str.contains("Awake", na=False)][feature].values
    sleep = df_rec[df_rec["state"].str.contains("Sleep", na=False)][feature].values
    anes = df_rec[df_rec["state"].str.contains("Anesthetized", na=False)][feature].values
    
    if len(awake) == 0 or len(sleep) == 0 or len(anes) == 0:
        print("Warning: Missing data for one or more states. Cannot complete check.")
        return

    # Define groups for iteration
    states = [
        ("Awake", awake, "#4C9FD1"), 
        ("Sleep", sleep, "#F39C12"), 
        ("Anesthetized", anes, "#27AE60")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, data, color) in enumerate(states):
        # Applying Shapiro-Wilk Test
        stat, p_val = shapiro(data)
        status = "Passed (Normal)" if p_val >= 0.05 else "Failed (Not Normal)"
        
        print(f"{name:<15} | Shapiro p-value: {p_val:.4e} | {status}")

        # DYNAMIC OUTLIER CHECK (Triggers only on failure)
        if p_val < 0.05:
            print(f"    [!] {name} failed normality. Identifying potential outliers...")
            
            # Isolate the data just for this specific failing state
            state_data = df_rec[df_rec["state"].str.contains(name, na=False)]
            
            # Sort to find extreme highs and lows
            highest = state_data.sort_values(by=feature, ascending=False).head(3)
            lowest = state_data.sort_values(by=feature, ascending=True).head(3)
            
            print("        Top 3 HIGHEST values:")
            for _, row in highest.iterrows():
                print(f"          - {row['recording']}: {row[feature]:.4f}")
                
            print("        Top 3 LOWEST values:")
            for _, row in lowest.iterrows():
                print(f"          - {row['recording']}: {row[feature]:.4f}")
            print("    " + "-"*50)

        
        # The Quantile-Quantile Plot
        probplot(data, dist="norm", plot=axes[i])
        
        axes[i].set_title(f"{feature} Q-Q plot for {name} in {region} region\n(p = {p_val:.2e})", fontsize=12)
        axes[i].set_xlabel("Theoretical Quantiles", fontsize=12)
        axes[i].set_ylabel("Ordered Values", fontsize=12)
        
        scatter_line = axes[i].get_lines()[0]
        scatter_line.set_markerfacecolor(color)
        scatter_line.set_markeredgecolor('black')
        scatter_line.set_markersize(7)
        scatter_line.set_alpha(0.8)
        
        # Customize the theoretical 45-degree line
        trend_line = axes[i].get_lines()[1]
        trend_line.set_color('black')
        trend_line.set_linewidth(2)
        trend_line.set_linestyle('--')

    plt.suptitle(f"Normality Diagnostics: {feature.replace('_', ' ').title()} in {region}", 
                 fontsize=18, fontweight='bold', y=1.05)
    
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    plt.show()



df_all = pd.read_csv("ECoG_Granular_Results.csv")


# df_all = df_all[df_all['recording'] != 'Recording_18'] # dropping the outlier
df_all['state'] = df_all['state'].str.replace('_Region_State', '', regex=False)
df_all['state'] = df_all['state'].replace({'Anaesthetized': 'Anesthetized'})


all_regions = df_all['region'].unique()
print(f"Discovered {len(all_regions)} regions: {all_regions}")

for current_region in all_regions:
    check_normality_assumptions(df=df_all, region=current_region, feature="max_scale")

for current_region in all_regions:
    check_normality_assumptions(df=df_all, region=current_region, feature="bandwidth")
