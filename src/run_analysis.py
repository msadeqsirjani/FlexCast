"""
Complete Analysis Runner
Executes both model performance visualization and data diagnostics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from visualize_results import main as visualize_main
from diagnostic_analysis import main as diagnostic_main


def main():
    """Run complete analysis pipeline"""
    print("\n" + "=" * 80)
    print("FLEXTRACK CHALLENGE - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()

    # Run model performance visualization
    print("🔍 STEP 1: Analyzing Model Performance")
    print("-" * 80)
    try:
        visualize_main()
        print("\n✓ Model performance analysis complete!")
    except Exception as e:
        print(f"\n⚠️  Error in performance analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

    # Run data diagnostics
    print("🔍 STEP 2: Running Data Diagnostics")
    print("-" * 80)
    try:
        diagnostic_main()
        print("\n✓ Data diagnostics complete!")
    except Exception as e:
        print(f"\n⚠️  Error in diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ COMPLETE ANALYSIS FINISHED")
    print("=" * 80)
    print("\nGenerated outputs:")
    print("  📊 Visualizations: ../results/figures/")
    print("  📄 Reports: ../results/")
    print("\nKey files to review:")
    print("  • performance_analysis_report.txt - Model comparison summary")
    print("  • classification_comparison.png - Classification metrics")
    print("  • regression_comparison.png - Regression metrics")
    print("  • target_distributions.png - Target variable analysis")
    print("  • class_imbalance.png - Class balance analysis")
    print("  • temporal_patterns.png - Time-based patterns")
    print()


if __name__ == "__main__":
    main()
