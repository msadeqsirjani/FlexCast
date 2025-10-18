"""
Data Distribution and Diagnostic Analysis
Identifies potential issues causing cross-site performance degradation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_loader import DataLoader
from feature_engineering import FeatureEngineer

sns.set_style("whitegrid")


class DataDiagnostics:
    """Comprehensive data diagnostics for cross-site analysis"""

    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.loader = DataLoader(data_dir)
        self.engineer = FeatureEngineer()

    def analyze_target_distributions(self, version: str = "v0.2"):
        """Analyze target variable distributions across sites"""
        print("\n" + "=" * 80)
        print("TARGET DISTRIBUTION ANALYSIS")
        print("=" * 80)

        # Load data
        data = self.loader.load_training_data(version=version)

        sites = data['Site'].unique()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Target Variable Distributions Across Sites', fontsize=16, fontweight='bold')

        results = {}

        for idx, site in enumerate(sites):
            site_data = data[data['Site'] == site]

            # Classification target
            ax_class = axes[0, idx]
            class_counts = site_data['Demand_Response_Flag'].value_counts().sort_index()
            class_counts.plot(kind='bar', ax=ax_class, color=['red', 'gray', 'green'])
            ax_class.set_title(f'{site.upper()} - Classification Target', fontweight='bold')
            ax_class.set_xlabel('DR Flag (-1, 0, 1)')
            ax_class.set_ylabel('Count')
            ax_class.grid(True, alpha=0.3)

            # Add percentage labels
            total = class_counts.sum()
            for i, v in enumerate(class_counts.values):
                ax_class.text(i, v, f'{v}\n({v/total*100:.1f}%)',
                            ha='center', va='bottom', fontsize=9)

            # Regression target
            ax_reg = axes[1, idx]
            site_data['Demand_Response_Capacity_kW'].hist(bins=50, ax=ax_reg, edgecolor='black')
            ax_reg.set_title(f'{site.upper()} - Regression Target', fontweight='bold')
            ax_reg.set_xlabel('DR Capacity (kW)')
            ax_reg.set_ylabel('Frequency')
            ax_reg.axvline(site_data['Demand_Response_Capacity_kW'].mean(),
                          color='red', linestyle='--', label='Mean')
            ax_reg.axvline(site_data['Demand_Response_Capacity_kW'].median(),
                          color='green', linestyle='--', label='Median')
            ax_reg.legend()
            ax_reg.grid(True, alpha=0.3)

            # Store statistics
            results[site] = {
                'classification': {
                    'class_distribution': class_counts.to_dict(),
                    'imbalance_ratio': class_counts.max() / class_counts.min()
                },
                'regression': {
                    'mean': site_data['Demand_Response_Capacity_kW'].mean(),
                    'median': site_data['Demand_Response_Capacity_kW'].median(),
                    'std': site_data['Demand_Response_Capacity_kW'].std(),
                    'min': site_data['Demand_Response_Capacity_kW'].min(),
                    'max': site_data['Demand_Response_Capacity_kW'].max(),
                    'q25': site_data['Demand_Response_Capacity_kW'].quantile(0.25),
                    'q75': site_data['Demand_Response_Capacity_kW'].quantile(0.75)
                }
            }

        plt.tight_layout()
        save_path = Path("../results/figures")
        save_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path / 'target_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path / 'target_distributions.png'}")
        plt.close()

        # Print summary statistics
        self._print_target_summary(results)

        return results

    def _print_target_summary(self, results):
        """Print target distribution summary"""
        print("\n" + "-" * 80)
        print("CLASSIFICATION TARGET SUMMARY")
        print("-" * 80)
        print(f"{'Site':<10} {'Class -1':<12} {'Class 0':<12} {'Class 1':<12} {'Imbalance':<12}")
        print("-" * 80)

        for site, stats in results.items():
            dist = stats['classification']['class_distribution']
            imb = stats['classification']['imbalance_ratio']
            print(f"{site:<10} {dist.get(-1, 0):<12} {dist.get(0, 0):<12} "
                  f"{dist.get(1, 0):<12} {imb:<12.2f}")

        print("\n" + "-" * 80)
        print("REGRESSION TARGET SUMMARY")
        print("-" * 80)
        print(f"{'Site':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Range':<12}")
        print("-" * 80)

        for site, stats in results.items():
            reg = stats['regression']
            range_val = reg['max'] - reg['min']
            print(f"{site:<10} {reg['mean']:<12.4f} {reg['std']:<12.4f} "
                  f"{reg['min']:<12.4f} {reg['max']:<12.4f} {range_val:<12.4f}")

        # Identify scaling issues
        print("\n⚠️  POTENTIAL SCALING ISSUES:")
        means = [stats['regression']['mean'] for stats in results.values()]
        stds = [stats['regression']['std'] for stats in results.values()]
        ranges = [stats['regression']['max'] - stats['regression']['min']
                 for stats in results.values()]

        if max(means) / min(means) > 5:
            print(f"  • Mean target values vary by {max(means)/min(means):.1f}x across sites")
            print(f"    → Consider per-site normalization")

        if max(stds) / min(stds) > 5:
            print(f"  • Standard deviations vary by {max(stds)/min(stds):.1f}x across sites")
            print(f"    → Different data variance across sites")

        if max(ranges) / min(ranges) > 10:
            print(f"  • Value ranges vary by {max(ranges)/min(ranges):.1f}x across sites")
            print(f"    → Possible different measurement scales")

    def analyze_feature_distributions(self, version: str = "v0.2", top_features: int = 10):
        """Analyze feature distributions across sites"""
        print("\n" + "=" * 80)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 80)

        data = self.loader.load_training_data(version=version)

        # Core numerical features to analyze
        core_features = ['Building_Power_kW', 'Dry_Bulb_Temperature_C',
                        'Global_Horizontal_Radiation_W/m2']

        sites = data['Site'].unique()
        fig, axes = plt.subplots(len(core_features), len(sites), figsize=(15, 12))
        fig.suptitle('Feature Distributions Across Sites', fontsize=16, fontweight='bold')

        for feat_idx, feature in enumerate(core_features):
            for site_idx, site in enumerate(sites):
                ax = axes[feat_idx, site_idx]
                site_data = data[data['Site'] == site]

                site_data[feature].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'{site.upper()} - {feature}', fontsize=9)
                ax.set_xlabel(feature, fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

                # Add mean line
                mean_val = site_data[feature].mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1)

        plt.tight_layout()
        save_path = Path("../results/figures")
        plt.savefig(save_path / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path / 'feature_distributions.png'}")
        plt.close()

        # Statistical comparison
        self._compare_feature_statistics(data, core_features)

    def _compare_feature_statistics(self, data, features):
        """Compare feature statistics across sites"""
        print("\n" + "-" * 80)
        print("FEATURE STATISTICS COMPARISON")
        print("-" * 80)

        for feature in features:
            print(f"\n{feature}:")
            print(f"{'Site':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print("-" * 60)

            for site in data['Site'].unique():
                site_data = data[data['Site'] == site][feature]
                print(f"{site:<10} {site_data.mean():<12.4f} {site_data.std():<12.4f} "
                      f"{site_data.min():<12.4f} {site_data.max():<12.4f}")

    def check_class_imbalance(self, version: str = "v0.2"):
        """Check and visualize class imbalance"""
        print("\n" + "=" * 80)
        print("CLASS IMBALANCE ANALYSIS")
        print("=" * 80)

        data = self.loader.load_training_data(version=version)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Class Imbalance Analysis', fontsize=16, fontweight='bold')

        # Overall class distribution
        ax1 = axes[0]
        overall_dist = data['Demand_Response_Flag'].value_counts().sort_index()
        colors = ['#e74c3c', '#95a5a6', '#2ecc71']
        overall_dist.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Overall Class Distribution', fontweight='bold')
        ax1.set_xlabel('DR Flag')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['-1 (Decrease)', '0 (No change)', '1 (Increase)'], rotation=0)

        total = overall_dist.sum()
        for i, v in enumerate(overall_dist.values):
            ax1.text(i, v, f'{v:,}\n({v/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

        # Per-site comparison
        ax2 = axes[1]
        site_class_data = []
        for site in data['Site'].unique():
            site_dist = data[data['Site'] == site]['Demand_Response_Flag'].value_counts()
            site_total = site_dist.sum()
            for cls in [-1, 0, 1]:
                site_class_data.append({
                    'Site': site,
                    'Class': cls,
                    'Percentage': (site_dist.get(cls, 0) / site_total) * 100
                })

        df_imbalance = pd.DataFrame(site_class_data)
        df_pivot = df_imbalance.pivot(index='Site', columns='Class', values='Percentage')
        df_pivot.plot(kind='bar', ax=ax2, color=colors, stacked=True)
        ax2.set_title('Class Distribution by Site (%)', fontweight='bold')
        ax2.set_xlabel('Site')
        ax2.set_ylabel('Percentage')
        ax2.legend(title='DR Flag', labels=['-1 (Decrease)', '0 (No change)', '1 (Increase)'])
        ax2.set_ylim([0, 100])

        plt.tight_layout()
        save_path = Path("../results/figures")
        plt.savefig(save_path / 'class_imbalance.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path / 'class_imbalance.png'}")
        plt.close()

        # Calculate imbalance metrics
        print("\nClass Imbalance Metrics:")
        print("-" * 60)
        minority_class = overall_dist.min()
        majority_class = overall_dist.max()
        imbalance_ratio = majority_class / minority_class
        print(f"Imbalance Ratio (majority/minority): {imbalance_ratio:.2f}")

        if imbalance_ratio > 3:
            print("⚠️  SEVERE IMBALANCE DETECTED")
            print("   Recommendations:")
            print("   • Apply SMOTE or other oversampling techniques")
            print("   • Use class weights in model training")
            print("   • Consider ensemble methods with balanced sampling")

    def analyze_temporal_patterns(self, version: str = "v0.2"):
        """Analyze temporal patterns in demand response"""
        print("\n" + "=" * 80)
        print("TEMPORAL PATTERN ANALYSIS")
        print("=" * 80)

        data = self.loader.load_training_data(version=version)
        # Timestamp_Local is already a datetime from data_loader
        data['Hour'] = data['Timestamp_Local'].dt.hour
        data['DayOfWeek'] = data['Timestamp_Local'].dt.dayofweek

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Temporal Patterns in Demand Response', fontsize=16, fontweight='bold')

        # Hourly patterns
        ax1 = axes[0, 0]
        for site in data['Site'].unique():
            site_data = data[data['Site'] == site]
            hourly_mean = site_data.groupby('Hour')['Demand_Response_Capacity_kW'].mean()
            ax1.plot(hourly_mean.index, hourly_mean.values, marker='o', label=site, linewidth=2)

        ax1.set_title('Average DR Capacity by Hour', fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average DR Capacity (kW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Day of week patterns
        ax2 = axes[0, 1]
        for site in data['Site'].unique():
            site_data = data[data['Site'] == site]
            dow_mean = site_data.groupby('DayOfWeek')['Demand_Response_Capacity_kW'].mean()
            ax2.plot(dow_mean.index, dow_mean.values, marker='s', label=site, linewidth=2)

        ax2.set_title('Average DR Capacity by Day of Week', fontweight='bold')
        ax2.set_xlabel('Day of Week (0=Mon, 6=Sun)')
        ax2.set_ylabel('Average DR Capacity (kW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # DR Flag frequency by hour
        ax3 = axes[1, 0]
        dr_hourly = data.groupby(['Hour', 'Demand_Response_Flag']).size().unstack(fill_value=0)
        dr_hourly.plot(kind='bar', stacked=False, ax=ax3, color=['red', 'gray', 'green'])
        ax3.set_title('DR Flag Frequency by Hour', fontweight='bold')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('Count')
        ax3.legend(title='DR Flag')

        # Building power vs DR capacity
        ax4 = axes[1, 1]
        for site in data['Site'].unique():
            site_data = data[data['Site'] == site].sample(min(1000, len(data)))
            ax4.scatter(site_data['Building_Power_kW'],
                       site_data['Demand_Response_Capacity_kW'],
                       alpha=0.5, label=site, s=10)

        ax4.set_title('Building Power vs DR Capacity', fontweight='bold')
        ax4.set_xlabel('Building Power (kW)')
        ax4.set_ylabel('DR Capacity (kW)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = Path("../results/figures")
        plt.savefig(save_path / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path / 'temporal_patterns.png'}")
        plt.close()

    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE DIAGNOSTIC REPORT")
        print("=" * 80)

        target_stats = self.analyze_target_distributions()
        self.analyze_feature_distributions()
        self.check_class_imbalance()
        self.analyze_temporal_patterns()

        print("\n" + "=" * 80)
        print("✓ DIAGNOSTIC ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nAll visualizations saved to: ../results/figures/")
        print("Review the plots to understand:")
        print("  1. Target distribution differences across sites")
        print("  2. Feature scaling issues")
        print("  3. Class imbalance severity")
        print("  4. Temporal patterns in demand response")


def main():
    """Run complete diagnostic analysis"""
    diagnostics = DataDiagnostics()
    diagnostics.generate_diagnostic_report()


if __name__ == "__main__":
    main()
