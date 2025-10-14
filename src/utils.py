"""
Utility functions for the Energy Flexibility project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json
import warnings
warnings.filterwarnings('ignore')


def save_predictions(
    predictions: np.ndarray,
    timestamps: pd.Series,
    sites: pd.Series,
    filepath: str,
    include_capacity: bool = False,
    capacity_predictions: np.ndarray = None
):
    """
    Save predictions to CSV file

    Args:
        predictions: DR flag predictions
        timestamps: Timestamp series
        sites: Site series
        filepath: Output file path
        include_capacity: Whether to include capacity predictions
        capacity_predictions: DR capacity predictions
    """
    df = pd.DataFrame({
        'Site': sites,
        'Timestamp_Local': timestamps,
        'Demand_Response_Flag': predictions
    })

    if include_capacity and capacity_predictions is not None:
        df['Demand_Response_Capacity_kW'] = capacity_predictions

    df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")


def load_predictions(filepath: str) -> pd.DataFrame:
    """
    Load predictions from CSV file

    Args:
        filepath: Path to predictions file

    Returns:
        DataFrame with predictions
    """
    return pd.read_csv(filepath)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    figsize: tuple = (10, 8),
    save_path: str = None
):
    """
    Plot feature importance

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        figsize: Figure size
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=figsize)
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    task: str = 'regression',
    figsize: tuple = (12, 5),
    save_path: str = None
):
    """
    Plot predictions vs actual values

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        task: 'classification' or 'regression'
        figsize: Figure size
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if task == 'regression':
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Predictions vs Actual')
        axes[0].grid(True, alpha=0.3)

        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
    else:  # classification
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Confusion Matrix')

        # Class distribution
        unique, counts_true = np.unique(y_true, return_counts=True)
        _, counts_pred = np.unique(y_pred, return_counts=True)

        x = np.arange(len(unique))
        width = 0.35

        axes[1].bar(x - width/2, counts_true, width, label='Actual', alpha=0.8)
        axes[1].bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.8)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Class Distribution')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(unique)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_time_series(
    data: pd.DataFrame,
    columns: List[str],
    title: str = "Time Series",
    figsize: tuple = (15, 8),
    save_path: str = None
):
    """
    Plot time series data

    Args:
        data: DataFrame with timestamp and values
        columns: Columns to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(len(columns), 1, figsize=figsize)

    if len(columns) == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        axes[i].plot(data['Timestamp_Local'], data[col])
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)

        if i < len(columns) - 1:
            axes[i].set_xticklabels([])

    axes[-1].set_xlabel('Time')
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def save_results_json(results: Dict, filepath: str):
    """
    Save results to JSON file

    Args:
        results: Results dictionary
        filepath: Output file path
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results_converted = convert_numpy(results)

    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"Results saved to {filepath}")


def load_results_json(filepath: str) -> Dict:
    """
    Load results from JSON file

    Args:
        filepath: Path to results file

    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_summary_report(
    model_results: Dict[str, Dict],
    output_path: str,
    task: str = 'both'
):
    """
    Create a summary report of model results

    Args:
        model_results: Dictionary mapping model names to results
        output_path: Path to save report
        task: 'classification', 'regression', or 'both'
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MODEL COMPARISON REPORT")
    report_lines.append("Energy Flexibility and Demand Response Challenge")
    report_lines.append("="*80)
    report_lines.append("")

    if task in ['classification', 'both']:
        report_lines.append("\nCLASSIFICATION RESULTS")
        report_lines.append("-"*80)
        report_lines.append(f"{'Model':<20} {'F1 Score':<15} {'Geometric Mean':<15}")
        report_lines.append("-"*80)

        for model_name, results in sorted(model_results.items()):
            if 'classification' in results or 'f1_score_macro' in results:
                res = results.get('classification', results)
                f1 = res.get('f1_score_macro', 0)
                gm = res.get('geometric_mean_score', 0)
                report_lines.append(f"{model_name:<20} {f1:<15.4f} {gm:<15.4f}")

    if task in ['regression', 'both']:
        report_lines.append("\n\nREGRESSION RESULTS")
        report_lines.append("-"*80)
        report_lines.append(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'NMAE(%)':<12} {'CV-RMSE(%)':<12}")
        report_lines.append("-"*80)

        for model_name, results in sorted(model_results.items()):
            if 'regression' in results or 'mae' in results:
                res = results.get('regression', results)
                mae = res.get('mae', 0)
                rmse = res.get('rmse', 0)
                nmae = res.get('nmae_range', 0)
                cv_rmse = res.get('cv_rmse', 0)
                report_lines.append(f"{model_name:<20} {mae:<12.4f} {rmse:<12.4f} {nmae:<12.2f} {cv_rmse:<12.2f}")

    report_lines.append("\n" + "="*80)

    report_text = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    print("Utility functions for Energy Flexibility project")
