#!/usr/bin/env python3
"""
Data Analysis Visualization Script
Generates ACM-style figures for statistical analysis of FlexTrack data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ACM paper style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
})

# Color palette for professional visualization
COLORS = {
    "siteA": "#4A90E2",
    "siteB": "#E85D75",
    "siteC": "#50C878",
    "positive": "#E85D75",
    "negative": "#4A90E2",
    "neutral": "#95A5A6",
}

SITES = ["siteA", "siteB", "siteC"]


def setup_dirs():
    """Create output directories for figures"""
    for d in [Path("figures"), Path("../docs/figures")]:
        d.mkdir(exist_ok=True, parents=True)
    return Path("figures"), Path("../docs/figures")


def save_fig(fig, name):
    """Save figure to both figures and docs directories"""
    out_dir, docs_dir = setup_dirs()
    for d in [out_dir, docs_dir]:
        fig.savefig(d / name, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {name}")
    plt.close()


def load_training_data(data_path="../data/97d68ae9-197d-4968-a5dd-ff50f8956f59_flextrack-2025-training-data-v0.2.csv"):
    """Load and preprocess training data"""
    df = pd.read_csv(data_path)
    df["Timestamp_Local"] = pd.to_datetime(df["Timestamp_Local"])
    df["Hour"] = df["Timestamp_Local"].dt.hour
    df["Month"] = df["Timestamp_Local"].dt.month
    df["DayOfWeek"] = df["Timestamp_Local"].dt.dayofweek
    return df


def plot_class_imbalance(df):
    """
    Figure 1: Class Imbalance Analysis
    Shows distribution of Demand Response Flag across sites
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

    for idx, site in enumerate(SITES):
        site_data = df[df["Site"] == site]
        counts = site_data["Demand_Response_Flag"].value_counts().sort_index()

        # Ensure both classes are present (reindex to include 0 and 1)
        counts = counts.reindex([0, 1], fill_value=0)

        # Calculate percentages
        total = len(site_data)
        percentages = (counts / total * 100).values

        # Create bars
        bars = axes[idx].bar(
            [0, 1],
            counts.values,
            width=0.6,
            color=[COLORS["negative"], COLORS["positive"]],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8
        )

        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{pct:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9
            )

        axes[idx].set_title(f"Site {site[-1].upper()}", fontweight="normal", fontsize=12)
        axes[idx].set_xlabel("DR Flag", fontsize=11)
        axes[idx].set_ylabel("Count", fontsize=11)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(["No DR (0)", "DR (1)"], fontsize=9)
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        axes[idx].set_facecolor("white")

        # Add grid
        axes[idx].yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        axes[idx].set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, "data_class_imbalance.png")


def plot_feature_distributions(df):
    """
    Figure 2: Feature Distribution Across Sites
    Box plots showing distribution of key features
    """
    features = [
        ("Dry_Bulb_Temperature_C", "Temperature (°C)"),
        ("Global_Horizontal_Radiation_W/m2", "Solar Radiation (W/m²)"),
        ("Building_Power_kW", "Building Power (kW)")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, (col, label) in enumerate(features):
        data_by_site = [df[df["Site"] == site][col].values for site in SITES]

        bp = axes[idx].boxplot(
            data_by_site,
            tick_labels=[s[-1].upper() for s in SITES],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            medianprops=dict(color=COLORS["positive"], linewidth=1.5),
            flierprops=dict(marker='o', markersize=2, alpha=0.3, markerfacecolor='gray')
        )

        # Color boxes by site
        for patch, site in zip(bp['boxes'], SITES):
            patch.set_facecolor(COLORS[site])
            patch.set_alpha(0.6)

        axes[idx].set_xlabel("Site", fontsize=11)
        axes[idx].set_ylabel(label, fontsize=11)
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        axes[idx].set_facecolor("white")
        axes[idx].yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        axes[idx].set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, "data_feature_distributions.png")


def plot_temporal_patterns(df):
    """
    Figure 4: Temporal Patterns
    Shows average DR capacity and flag by hour of day
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.5))

    # Plot 1: Average DR Capacity by Hour
    for site in SITES:
        site_data = df[df["Site"] == site]
        hourly_avg = site_data.groupby("Hour")["Demand_Response_Capacity_kW"].mean()
        ax1.plot(hourly_avg.index, hourly_avg.values,
                marker='o', markersize=3, linewidth=1.5,
                label=f"Site {site[-1].upper()}",
                color=COLORS[site], alpha=0.85)

    ax1.set_xlabel("Hour of Day", fontsize=11)
    ax1.set_ylabel("Avg DR Capacity (kW)", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_facecolor("white")
    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xlim([-0.5, 23.5])
    ax1.set_xticks(range(0, 24, 3))

    # Plot 2: DR Flag Activation Rate by Hour
    for site in SITES:
        site_data = df[df["Site"] == site]
        hourly_rate = site_data.groupby("Hour")["Demand_Response_Flag"].mean() * 100
        ax2.plot(hourly_rate.index, hourly_rate.values,
                marker='s', markersize=3, linewidth=1.5,
                label=f"Site {site[-1].upper()}",
                color=COLORS[site], alpha=0.85)

    ax2.set_xlabel("Hour of Day", fontsize=11)
    ax2.set_ylabel("DR Activation Rate (%)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_facecolor("white")
    ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xlim([-0.5, 23.5])
    ax2.set_xticks(range(0, 24, 3))

    # Add single legend at upper center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               frameon=True, fancybox=False, edgecolor="black", framealpha=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "data_temporal_patterns.png")


def plot_target_distribution(df):
    """
    Figure 5: Target Variable Distribution
    Distribution of DR Capacity across sites (only non-zero values)
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(SITES):
        site_data = df[df["Site"] == site]

        # Separate positive and negative DR capacity
        positive_dr = site_data[site_data["Demand_Response_Capacity_kW"] > 0]["Demand_Response_Capacity_kW"]
        negative_dr = site_data[site_data["Demand_Response_Capacity_kW"] < 0]["Demand_Response_Capacity_kW"]

        # Create histogram
        if len(positive_dr) > 0:
            axes[idx].hist(positive_dr, bins=30, alpha=0.7,
                          color=COLORS["positive"], edgecolor='black',
                          linewidth=0.5, label='Positive')

        if len(negative_dr) > 0:
            axes[idx].hist(negative_dr, bins=30, alpha=0.7,
                          color=COLORS["negative"], edgecolor='black',
                          linewidth=0.5, label='Negative')

        axes[idx].set_title(f"Site {site[-1].upper()}", fontweight="normal", fontsize=12)
        axes[idx].set_xlabel("DR Capacity (kW)", fontsize=11)
        axes[idx].set_ylabel("Frequency", fontsize=11)
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        axes[idx].set_facecolor("white")
        axes[idx].yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        axes[idx].set_axisbelow(True)

    # Add single legend at upper center
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # Only add legend if there are items to show
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
                   frameon=True, fancybox=False, edgecolor="black", framealpha=1)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "data_target_distribution.png")


def plot_monthly_statistics(df):
    """
    Figure 6: Monthly Statistics
    Shows seasonal patterns in DR events and capacity
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.5))

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot 1: DR Event Count by Month
    for site in SITES:
        site_data = df[df["Site"] == site]
        monthly_events = site_data[site_data["Demand_Response_Flag"] == 1].groupby("Month").size()
        # Reindex to ensure all months are present
        monthly_events = monthly_events.reindex(range(1, 13), fill_value=0)

        ax1.plot(monthly_events.index, monthly_events.values,
                marker='o', markersize=4, linewidth=1.5,
                label=f"Site {site[-1].upper()}",
                color=COLORS[site], alpha=0.85)

    ax1.set_xlabel("Month", fontsize=11)
    ax1.set_ylabel("DR Event Count", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_facecolor("white")
    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names, fontsize=9)
    ax1.set_xlim([0.5, 12.5])

    # Plot 2: Average Building Power by Month
    for site in SITES:
        site_data = df[df["Site"] == site]
        monthly_power = site_data.groupby("Month")["Building_Power_kW"].mean()
        monthly_power = monthly_power.reindex(range(1, 13), fill_value=0)

        ax2.plot(monthly_power.index, monthly_power.values,
                marker='s', markersize=4, linewidth=1.5,
                label=f"Site {site[-1].upper()}",
                color=COLORS[site], alpha=0.85)

    ax2.set_xlabel("Month", fontsize=11)
    ax2.set_ylabel("Avg Building Power (kW)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_facecolor("white")
    ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names, fontsize=9)
    ax2.set_xlim([0.5, 12.5])

    # Add single legend at upper center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               frameon=True, fancybox=False, edgecolor="black", framealpha=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "data_monthly_statistics.png")


def main():
    """Generate all data analysis figures"""
    print("Loading data...")
    df = load_training_data()

    print(f"Data shape: {df.shape}")
    print(f"Sites: {df['Site'].unique()}")
    print(f"Date range: {df['Timestamp_Local'].min()} to {df['Timestamp_Local'].max()}\n")

    print("Generating figures...")
    plot_class_imbalance(df)
    plot_feature_distributions(df)
    plot_temporal_patterns(df)
    plot_target_distribution(df)
    plot_monthly_statistics(df)

    print("\nAll 5 figures generated successfully!")
    print("Figures saved to: ./figures/ and ../docs/figures/")


if __name__ == "__main__":
    main()
