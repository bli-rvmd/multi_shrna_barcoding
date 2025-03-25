####
# utility functions of multi shrna screening project
####

import pandas as pd
import numpy as np
from prettytable import PrettyTable
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import pearsonr

from matplotlib_venn import venn3
import matplotlib.patches as patches
from adjustText import adjust_text

def test_fn():
    return np.log2(8)

## utility function of printing table
def ViewTable(df, top_n_rows = None):
    table = PrettyTable(df.columns.tolist())
    if top_n_rows:
        df_tmp = df.head(top_n_rows)
    else:
        df_tmp = df
    for row in df_tmp.itertuples(index=False, name=None):
        table.add_row(row)
    print(table)


# function for bootstrapped confidence interval estimation of median
def bootstrap_median_ci(data, num_resamples=1000, ci=95):
    boot_medians = [np.median(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_resamples)]
    lower_bound = np.percentile(boot_medians, (100 - ci) / 2)
    upper_bound = np.percentile(boot_medians, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound


## function to generate scatter plot of median and its bootstrapped CI for RTN (relative cell/tumor number) for each experimental condition or predictive effect of RTN_A / RTN_B, where A,B correspond to treatment, vehicle groups under a particular timepoint and dosage 
def plot_bootstrapped_scatter(df_summary, x_timepoint, x_dosage, y_timepoint, y_dosage, metric):
    """
    Generates a scatter plot of bootstrapped median with CI for RTN_A, RTN_B, or Predictive_Effect.
    
    Parameters:
    df_summary : DataFrame
        The summary dataframe containing median and bootstrapped 95% CI values.
    x_timepoint : str
        The timepoint for the x-axis.
    x_dosage : str
        The dosage for the x-axis.
    y_timepoint : str
        The timepoint for the y-axis.
    y_dosage : str
        The dosage for the y-axis.
    metric : str
        The metric to plot ("RTN_A", "RTN_B", or "Predictive_Effect").
    """
    
    # Filter the dataframe based on user-specified timepoints and dosages
    df_x = df_summary[(df_summary["Timepoint"] == x_timepoint) & (df_summary["Dosage"] == x_dosage)]
    df_y = df_summary[(df_summary["Timepoint"] == y_timepoint) & (df_summary["Dosage"] == y_dosage)]

    # Merge to ensure only common Target_Genes are included
    df_merged = df_x.merge(df_y, on="Target_Gene", suffixes=("_x", "_y"))

    # Select appropriate metric columns
    x_values = df_merged[f"{metric}_median_x"]
    y_values = df_merged[f"{metric}_median_y"]

    x_err_lower = df_merged[f"{metric}_median_x"] - df_merged[f"{metric}_CI_lower_x"]
    x_err_upper = df_merged[f"{metric}_CI_upper_x"] - df_merged[f"{metric}_median_x"]
    x_err = np.array([x_err_lower, x_err_upper])

    y_err_lower = df_merged[f"{metric}_median_y"] - df_merged[f"{metric}_CI_lower_y"]
    y_err_upper = df_merged[f"{metric}_CI_upper_y"] - df_merged[f"{metric}_median_y"]
    y_err = np.array([y_err_lower, y_err_upper])

    # Compute correlation coefficient
    r_value, _ = pearsonr(x_values.dropna(), y_values.dropna())

    # Define gene lists for annotation
    loss_of_representation_target_genes = ["Rpa1", "Rpa3", "Rps6", "Pcna", "Psmc5", "Rbx1", "Ran", "Snrpd1", "Rpl7", "Kif11"]
    neutral_control_target_genes = ["NT", "Trp53"]
    gain_of_representation_target_genes = ["Pten", "Kras(G12D)", "Kras(wt)"]

    # Plot scatter with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_values, y_values, xerr=x_err, yerr=y_err, fmt='o', ecolor='lightgray', capsize=3, alpha=0.8, label=f"R = {r_value:.2f}")

    # Extend diagonal reference line across the entire plot area
    min_plot, max_plot = plt.xlim()[0], plt.xlim()[1]  # Get current plot limits
    min_data, max_data = min(x_values.min(), y_values.min()), max(x_values.max(), y_values.max())
    min_range, max_range = min(min_plot, min_data), max(max_plot, max_data)
    plt.plot([min_range, max_range], [min_range, max_range], linestyle="dotted", color="black")

    # If Predictive_Effect is selected, add vertical and horizontal reference lines at zero
    if metric == "Predictive_Effect":
        plt.axhline(0, color="black", linestyle="dotted")
        plt.axvline(0, color="black", linestyle="dotted")


    # Annotate selected target genes with boxes and avoid overlap
    texts = []
    for i, txt in enumerate(df_merged["Target_Gene"]):
        if txt in loss_of_representation_target_genes:
            color = "blue"
        elif txt in neutral_control_target_genes:
            color = "green"
        elif txt in gain_of_representation_target_genes:
            color = "red"
        else:
            continue

        # Create text annotation inside a rectangle
        text = plt.text(x_values.iloc[i], y_values.iloc[i], txt, fontsize=9, color="black", ha='center', va='center', 
                        bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.3"))
        texts.append(text)

    # Adjust text positions to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Labels and title
    plt.xlabel(f"{metric} of {x_timepoint} ({x_dosage})")
    plt.ylabel(f"{metric} of {y_timepoint} ({y_dosage})")
    plt.title(f"{metric} Comparison: {x_timepoint} vs {y_timepoint}")
    plt.legend()
    plt.grid(True, linestyle="dotted", alpha=0.5)

    # Show plot
    plt.show()

    # # Example usage
    # plot_bootstrapped_scatter(df_summary_predictive_effect, "T10", "3.5nM", "T13", "3.5nM", "RTN_A")


## function to run 'plot_bootstrapped_scatter' in an optimized grid across time points
def plot_optimized_grid(df_summary, dosage, metric):
    """
    """
    timepoints = ["T7", "T10", "T13"]
    fig, axes = plt.subplots(len(timepoints), len(timepoints), figsize=(6, 6), constrained_layout=True)
    fig.suptitle(f"{metric} Pairwise Comparisons for {dosage} Dosage", fontsize=12)

    # Filter data for the specified dosage
    df_filtered = df_summary[df_summary["Dosage"] == dosage]

    for i, row_tp in enumerate(timepoints):
        for j, col_tp in enumerate(timepoints):
            ax = axes[i, j]

            if i > j:
                # Pairwise scatter plot
                df_x = df_filtered[df_filtered["Timepoint"] == col_tp]
                df_y = df_filtered[df_filtered["Timepoint"] == row_tp]
                df_merged = df_x.merge(df_y, on="Target_Gene", suffixes=("_x", "_y"))

                x_values = df_merged[f"{metric}_median_x"]
                y_values = df_merged[f"{metric}_median_y"]

                # Compute correlation coefficient
                if len(x_values.dropna()) > 1 and len(y_values.dropna()) > 1:
                    r_value, _ = pearsonr(x_values.dropna(), y_values.dropna())
                    ax.set_title(f"R = {r_value:.2f}", fontsize=8)
                else:
                    ax.set_title("R = N/A", fontsize=8)

                ax.scatter(x_values, y_values, alpha=0.6, s=5)
                ax.plot([x_values.min(), x_values.max()], [x_values.min(), x_values.max()], 'k--', alpha=0.5)

            elif i == j:
                # Group labels
                group_size = df_filtered[df_filtered["Timepoint"] == row_tp].shape[0]
                ax.text(0.5, 0.5, f"Group {row_tp}\n(N={group_size})", ha='center', va='center', fontsize=10, fontweight='bold')
                ax.set_frame_on(False)

            else:
                # Empty upper triangle
                ax.set_visible(False)

    plt.show()
