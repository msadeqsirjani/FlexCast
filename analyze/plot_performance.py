#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)

ALGO_ABBR = {
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CAT",
    "HistGradientBoosting": "HGB",
    "Ensemble": "ENS",
    "Cascade": "CAS",
}

COLORS = {
    "classification": "#4A90E2",
    "regression": "#E85D75",
    "macro": "#50C878",
    "micro": "#F39C12",
    "weighted": "#9B59B6",
    "gmean": "#16A085",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def setup_ax(ax, site):
    ax.set_title(f"Site {site[-1].upper()}", fontweight="normal", fontsize=12)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for y in [25, 50, 75, 100]:
        ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)


def plot_accuracy(results_dir="../results"):
    results_dir = Path(results_dir)
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    sites = ["siteA", "siteB", "siteC"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(sites):
        ax = axes[idx]
        class_file = results_dir / "classification" / "metrics" / f"results_{site}.json"
        reg_file = results_dir / "regression" / "metrics" / f"results_{site}.json"
        class_data = load_json(class_file) if class_file.exists() else {}
        reg_data = load_json(reg_file) if reg_file.exists() else {}

        algos, class_scores, reg_scores = [], [], []

        for key, val in class_data.items():
            algo = key.replace("_classification", "")
            if algo in ALGO_ABBR:
                algos.append(ALGO_ABBR[algo])
                class_scores.append(val["validation"]["accuracy"] * 100)

        reg_scores_dict = {}
        for key, val in reg_data.items():
            algo = key.replace("_regression", "")
            if algo in ALGO_ABBR:
                reg_scores_dict[ALGO_ABBR[algo]] = max(0, 1 - val["validation"].get("nmae_range", 1.0)) * 100

        reg_scores = [reg_scores_dict.get(algo, 0) for algo in algos]

        x = np.arange(len(algos))
        width = 0.35

        ax.bar(
            x - width/2,
            class_scores,
            width,
            label="Classification",
            color=COLORS["classification"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x + width/2,
            reg_scores,
            width,
            label="Regression",
            color=COLORS["regression"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.set_xlim([-0.6, len(algos) - 0.4])
        ax.set_ylim([0, 108])
        ax.set_ylabel("Performance Score (%)", fontsize=11)
        setup_ax(ax, site)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(output_dir / "accuracy.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_f1_scores(results_dir="../results"):
    results_dir = Path(results_dir)
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    sites = ["siteA", "siteB", "siteC"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(sites):
        ax = axes[idx]
        class_file = results_dir / "classification" / "metrics" / f"results_{site}.json"
        data = load_json(class_file) if class_file.exists() else {}

        algos, macro, micro, weighted = [], [], [], []

        for key, val in data.items():
            algo = key.replace("_classification", "")
            if algo in ALGO_ABBR:
                algos.append(ALGO_ABBR[algo])
                v = val["validation"]
                macro.append(v.get("f1_score_macro", 0) * 100)
                micro.append(v.get("f1_score_micro", 0) * 100)
                weighted.append(v.get("f1_score_weighted", 0) * 100)

        x = np.arange(len(algos))
        width = 0.23
        ax.bar(
            x - width,
            macro,
            width,
            label="Macro",
            color=COLORS["macro"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x,
            micro,
            width,
            label="Micro",
            color=COLORS["micro"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x + width,
            weighted,
            width,
            label="Weighted",
            color=COLORS["weighted"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.set_xlim([-0.6, len(algos) - 0.4])
        ax.set_ylim([0, 108])
        ax.set_ylabel("F1 Score (%)", fontsize=11)
        setup_ax(ax, site)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(output_dir / "f1_scores.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_gmean(results_dir="../results"):
    results_dir = Path(results_dir)
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    sites = ["siteA", "siteB", "siteC"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(sites):
        ax = axes[idx]
        class_file = results_dir / "classification" / "metrics" / f"results_{site}.json"
        data = load_json(class_file) if class_file.exists() else {}

        algos, gmeans = [], []

        for key, val in data.items():
            algo = key.replace("_classification", "")
            if algo in ALGO_ABBR:
                algos.append(ALGO_ABBR[algo])
                gmeans.append(val["validation"].get("geometric_mean_score", 0) * 100)

        ax.bar(
            range(len(algos)),
            gmeans,
            width=0.7,
            color=COLORS["gmean"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.set_xlim([-0.5, len(algos) - 0.5])
        ax.set_ylim([0, 108])
        ax.set_ylabel("G-Mean (%)", fontsize=11)
        setup_ax(ax, site)

    plt.tight_layout()
    plt.savefig(output_dir / "g_mean.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    plot_accuracy()
    plot_f1_scores()
    plot_gmean()
