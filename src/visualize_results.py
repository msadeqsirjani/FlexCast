"""
Visualization Script for Cross-Site Model Comparison
Analyzes and visualizes model performance across different sites
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_results(results_dir: str = "../results"):
    """Load results from all sites"""
    results_dir = Path(results_dir)

    all_results = {}
    for site_file in results_dir.glob("results_site*.json"):
        site = site_file.stem.replace("results_", "")
        with open(site_file, 'r') as f:
            all_results[site] = json.load(f)

    return all_results


def extract_metrics(all_results):
    """Extract metrics into DataFrames for easy plotting"""

    classification_data = []
    regression_data = []

    for site, results in all_results.items():
        for model_task, metrics in results.items():
            model_name = model_task.replace('_classification', '').replace('_regression', '')

            if 'classification' in model_task:
                classification_data.append({
                    'Site': site,
                    'Model': model_name,
                    'F1': metrics['validation'].get('f1_score_macro', 0),
                    'GM': metrics['validation'].get('geometric_mean_score', 0)
                })
            elif 'regression' in model_task:
                regression_data.append({
                    'Site': site,
                    'Model': model_name,
                    'MAE': metrics['validation'].get('mae', 0),
                    'RMSE': metrics['validation'].get('rmse', 0),
                    'NMAE': metrics['validation'].get('nmae', 0),
                    'CV-RMSE': metrics['validation'].get('cv_rmse', 0)
                })

    clf_df = pd.DataFrame(classification_data)
    reg_df = pd.DataFrame(regression_data)

    return clf_df, reg_df


def plot_classification_results(clf_df, save_path="../results/figures"):
    """Create comprehensive classification visualizations"""
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Classification Performance Across Sites', fontsize=16, fontweight='bold')

    # 1. F1 Score Comparison
    ax1 = axes[0, 0]
    clf_pivot = clf_df.pivot(index='Model', columns='Site', values='F1')
    clf_pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('F1 Score by Model and Site', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score (Macro)', fontsize=11)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.legend(title='Site', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(clf_df['F1'].max() * 1.2, 0.5)])

    # 2. Geometric Mean Comparison
    ax2 = axes[0, 1]
    gm_pivot = clf_df.pivot(index='Model', columns='Site', values='GM')
    gm_pivot.plot(kind='bar', ax=ax2, width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Geometric Mean by Model and Site', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Geometric Mean', fontsize=11)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.legend(title='Site', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(clf_df['GM'].max() * 1.2, 0.6)])

    # 3. Site-wise F1 Comparison
    ax3 = axes[1, 0]
    site_pivot = clf_df.pivot(index='Site', columns='Model', values='F1')
    site_pivot.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('F1 Score Across Sites (by Model)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1 Score (Macro)', fontsize=11)
    ax3.set_xlabel('Site', fontsize=11)
    ax3.legend(title='Model', fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. Model Stability (Coefficient of Variation across sites)
    ax4 = axes[1, 1]
    stability = clf_df.groupby('Model')['F1'].agg(['mean', 'std'])
    stability['cv'] = (stability['std'] / stability['mean']) * 100
    stability['cv'].plot(kind='bar', ax=ax4, color='coral')
    ax4.set_title('Model Stability (CV% of F1 across Sites)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax4.set_xlabel('Model', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% CV threshold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'classification_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path / 'classification_comparison.png'}")
    plt.close()

    return stability


def plot_regression_results(reg_df, save_path="../results/figures"):
    """Create comprehensive regression visualizations"""
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Regression Performance Across Sites', fontsize=16, fontweight='bold')

    # 1. MAE Comparison
    ax1 = axes[0, 0]
    mae_pivot = reg_df.pivot(index='Model', columns='Site', values='MAE')
    mae_pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('MAE by Model and Site', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error', fontsize=11)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.legend(title='Site', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale due to large variation

    # 2. RMSE Comparison
    ax2 = axes[0, 1]
    rmse_pivot = reg_df.pivot(index='Model', columns='Site', values='RMSE')
    rmse_pivot.plot(kind='bar', ax=ax2, width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('RMSE by Model and Site', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Root Mean Squared Error', fontsize=11)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.legend(title='Site', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale due to large variation

    # 3. Site Effect Visualization
    ax3 = axes[1, 0]
    site_mae = reg_df.groupby('Site')['MAE'].mean()
    colors = ['#2ecc71' if val < 0.5 else '#e74c3c' if val > 1 else '#f39c12'
              for val in site_mae.values]
    site_mae.plot(kind='bar', ax=ax3, color=colors)
    ax3.set_title('Average MAE per Site (Site Effect)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average MAE', fontsize=11)
    ax3.set_xlabel('Site', fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(site_mae.values):
        ax3.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    # 4. Model Performance Heatmap
    ax4 = axes[1, 1]
    heatmap_data = reg_df.pivot(index='Site', columns='Model', values='MAE')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax4,
                cbar_kws={'label': 'MAE'})
    ax4.set_title('MAE Heatmap (Site × Model)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Model', fontsize=11)
    ax4.set_ylabel('Site', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path / 'regression_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path / 'regression_comparison.png'}")
    plt.close()


def plot_cross_site_trends(clf_df, reg_df, save_path="../results/figures"):
    """Plot cross-site performance trends"""
    save_path = Path(save_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cross-Site Performance Degradation Analysis', fontsize=16, fontweight='bold')

    # Classification trends
    ax1 = axes[0]
    for model in clf_df['Model'].unique():
        model_data = clf_df[clf_df['Model'] == model].sort_values('Site')
        ax1.plot(model_data['Site'], model_data['F1'], marker='o', label=model, linewidth=2)

    ax1.set_title('Classification F1 Across Sites', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Site', fontsize=11)
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Regression trends
    ax2 = axes[1]
    for model in reg_df['Model'].unique():
        model_data = reg_df[reg_df['Model'] == model].sort_values('Site')
        ax2.plot(model_data['Site'], model_data['MAE'], marker='s', label=model, linewidth=2)

    ax2.set_title('Regression MAE Across Sites', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Site', fontsize=11)
    ax2.set_ylabel('MAE', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path / 'cross_site_trends.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path / 'cross_site_trends.png'}")
    plt.close()


def generate_summary_report(clf_df, reg_df, save_path="../results"):
    """Generate comprehensive summary report"""
    save_path = Path(save_path)

    report = []
    report.append("=" * 80)
    report.append("CROSS-SITE MODEL PERFORMANCE ANALYSIS")
    report.append("=" * 80)
    report.append("")

    # Classification Summary
    report.append("CLASSIFICATION PERFORMANCE SUMMARY")
    report.append("-" * 80)
    report.append("\nBest Model per Site (by F1 Score):")
    for site in clf_df['Site'].unique():
        site_data = clf_df[clf_df['Site'] == site]
        best = site_data.loc[site_data['F1'].idxmax()]
        report.append(f"  {site}: {best['Model']} (F1={best['F1']:.4f}, GM={best['GM']:.4f})")

    report.append("\nMost Stable Model (lowest CV across sites):")
    stability = clf_df.groupby('Model')['F1'].agg(['mean', 'std'])
    stability['cv'] = (stability['std'] / stability['mean']) * 100
    best_stable = stability['cv'].idxmin()
    report.append(f"  {best_stable} (CV={stability.loc[best_stable, 'cv']:.2f}%)")

    # Regression Summary
    report.append("\n" + "=" * 80)
    report.append("REGRESSION PERFORMANCE SUMMARY")
    report.append("-" * 80)
    report.append("\nBest Model per Site (by MAE):")
    for site in reg_df['Site'].unique():
        site_data = reg_df[reg_df['Site'] == site]
        best = site_data.loc[site_data['MAE'].idxmin()]
        report.append(f"  {site}: {best['Model']} (MAE={best['MAE']:.4f}, RMSE={best['RMSE']:.4f})")

    report.append("\nSite Performance Degradation (Average MAE):")
    site_mae = reg_df.groupby('Site')['MAE'].mean().sort_values()
    for site, mae in site_mae.items():
        report.append(f"  {site}: {mae:.4f}")

    # Identify problematic site
    worst_site = site_mae.idxmax()
    best_site = site_mae.idxmin()
    degradation_ratio = site_mae[worst_site] / site_mae[best_site]
    report.append(f"\n⚠️  Performance degradation from {best_site} to {worst_site}: {degradation_ratio:.1f}x")

    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)

    if degradation_ratio > 5:
        report.append("⚠️  CRITICAL: Severe cross-site performance degradation detected!")
        report.append("   • Check for label/feature scaling inconsistencies")
        report.append("   • Investigate data distribution shifts across sites")
        report.append("   • Consider site-specific normalization")

    if stability['cv'].max() > 15:
        report.append("⚠️  High model instability across sites detected")
        report.append("   • Consider domain adaptation techniques")
        report.append("   • Explore transfer learning approaches")

    if clf_df['F1'].mean() < 0.4:
        report.append("⚠️  Low classification performance overall")
        report.append("   • Check for class imbalance")
        report.append("   • Consider SMOTE or class weighting")
        report.append("   • Analyze feature importance per site")

    report.append("")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open(save_path / "performance_analysis_report.txt", 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {save_path / 'performance_analysis_report.txt'}")


def main():
    """Main execution"""
    print("Loading results...")
    all_results = load_results()

    print("Extracting metrics...")
    clf_df, reg_df = extract_metrics(all_results)

    print("\nGenerating visualizations...")
    stability = plot_classification_results(clf_df)
    plot_regression_results(reg_df)
    plot_cross_site_trends(clf_df, reg_df)

    print("\nGenerating summary report...")
    generate_summary_report(clf_df, reg_df)

    print("\n✓ Analysis complete!")
    print("  📊 Visualizations saved to: ../results/figures/")
    print("  📄 Report saved to: ../results/performance_analysis_report.txt")


if __name__ == "__main__":
    main()
