#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

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

SITES = ["siteA", "siteB", "siteC"]


def setup_dirs():
    for d in [Path("figures"), Path("../docs/figures")]:
        d.mkdir(exist_ok=True, parents=True)
    return Path("figures"), Path("../docs/figures")


def style_ax(ax, site, ylim, gridlines=None):
    ax.set_title(f"Site {site[-1].upper()}", fontweight="normal", fontsize=12)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(ylim)
    if gridlines:
        for y in gridlines:
            ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)


def save_fig(fig, name):
    out_dir, docs_dir = setup_dirs()
    for d in [out_dir, docs_dir]:
        fig.savefig(d / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def load_data(results_dir, task, site):
    path = Path(results_dir) / task / "metrics" / f"results_{site}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def plot_classification_accuracy(results_dir="../results"):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(SITES):
        data = load_data(results_dir, "classification", site)
        algos, scores = [], []

        for key, val in data.items():
            algo = key.replace("_classification", "")
            if algo in ALGO_ABBR:
                algos.append(ALGO_ABBR[algo])
                scores.append(val["validation"]["accuracy"] * 100)

        axes[idx].bar(range(len(algos)), scores, width=0.7, color=COLORS["classification"],
                      alpha=0.85, edgecolor="black", linewidth=0.8)
        axes[idx].set_xticks(range(len(algos)))
        axes[idx].set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        axes[idx].set_xlim([-0.5, len(algos) - 0.5])
        axes[idx].set_ylabel("Accuracy (%)", fontsize=11)
        style_ax(axes[idx], site, [60, 100], [65, 70, 75, 80, 85, 90, 95])

    plt.tight_layout()
    save_fig(fig, "classification_accuracy.png")


def plot_regression_performance(results_dir="../results"):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(SITES):
        data = load_data(results_dir, "regression", site)
        algos, scores = [], []

        for key, val in data.items():
            algo = key.replace("_regression", "")
            if algo in ALGO_ABBR:
                algos.append(ALGO_ABBR[algo])
                scores.append(val["validation"].get("rmse", 0))

        max_rmse = max(scores) if scores else 1.0
        ylim = [0, max_rmse * 1.1]

        axes[idx].bar(range(len(algos)), scores, width=0.7, color=COLORS["regression"],
                      alpha=0.85, edgecolor="black", linewidth=0.8)
        axes[idx].set_xticks(range(len(algos)))
        axes[idx].set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        axes[idx].set_xlim([-0.5, len(algos) - 0.5])
        axes[idx].set_ylabel("RMSE (kW)", fontsize=11)
        style_ax(axes[idx], site, ylim)

    plt.tight_layout()
    save_fig(fig, "regression_performance.png")


def plot_f1_scores(results_dir="../results"):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(SITES):
        data = load_data(results_dir, "classification", site)
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
        for offset, scores, label, color in [
            (-width, macro, "Macro", COLORS["macro"]),
            (0, micro, "Micro", COLORS["micro"]),
            (width, weighted, "Weighted", COLORS["weighted"])
        ]:
            axes[idx].bar(x + offset, scores, width, label=label, color=color,
                         alpha=0.85, edgecolor="black", linewidth=0.8)

        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        axes[idx].set_xlim([-0.6, len(algos) - 0.4])
        axes[idx].set_ylabel("F1 Score (%)", fontsize=11)
        style_ax(axes[idx], site, [0, 108], [25, 50, 75, 100])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               frameon=True, fancybox=False, edgecolor="black", framealpha=1)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, "f1_scores.png")


def plot_gmean(results_dir="../results"):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, site in enumerate(SITES):
        data = load_data(results_dir, "classification", site)
        algos, gmeans = [], []

        for key, val in data.items():
            algo = key.replace("_classification", "")
            if algo in ALGO_ABBR:
                algos.append(ALGO_ABBR[algo])
                gmeans.append(val["validation"].get("geometric_mean_score", 0) * 100)

        axes[idx].bar(range(len(algos)), gmeans, width=0.7, color=COLORS["gmean"],
                      alpha=0.85, edgecolor="black", linewidth=0.8)
        axes[idx].set_xticks(range(len(algos)))
        axes[idx].set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        axes[idx].set_xlim([-0.5, len(algos) - 0.5])
        axes[idx].set_ylabel("G-Mean (%)", fontsize=11)
        style_ax(axes[idx], site, [0, 108], [25, 50, 75, 100])

    plt.tight_layout()
    save_fig(fig, "g_mean.png")


if __name__ == "__main__":
    plot_classification_accuracy()
    plot_regression_performance()
    plot_f1_scores()
    plot_gmean()
