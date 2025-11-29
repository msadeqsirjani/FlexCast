# FlexCast - Energy Flexibility Forecasting

## Installation

```bash
git clone https://github.com/msadeqsirjani/FlexCast
cd FlexCast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

### 🚀 ALL OPTIMIZATIONS - Both Classification & Regression

```bash
python src/main.py --training-mode site-specific --tasks classification regression --sampler none --use-advanced-weights --use-ensemble --use-cascade --feature-selection --n-features 80 --data-dir data --output-dir results
```

**Includes:**
- Classification: Cascade, Ensemble, Advanced Weights, Feature Selection
- Regression: Ensemble, Advanced Weights, Feature Selection
- All 3 sites (siteA, siteB, siteC)

### All Optimizations - Classification Only

```bash
python src/main.py --training-mode site-specific --tasks classification --sampler none --use-advanced-weights --use-ensemble --use-cascade --feature-selection --n-features 80 --data-dir data --output-dir results
```

**Optimizations:**
- ✅ Cascade classifier (hierarchical)
- ✅ Ensemble voting (weighted average)
- ✅ Advanced class weights (effective samples)
- ✅ Feature selection (top 80)
- ✅ No resampling (best performance)

### All Optimizations - Regression Only

```bash
python src/main.py --training-mode site-specific --tasks regression --use-advanced-weights --use-ensemble --feature-selection --n-features 80 --data-dir data --output-dir results
```

**Optimizations:**
- ✅ Ensemble voting (MAE-weighted average)
- ✅ Advanced sample weights (quantile-based)
- ✅ Feature selection (top 80)

### Single Site Training

```bash
python src/main.py --site siteB --training-mode single --tasks classification
python src/main.py --site siteB --training-mode single --tasks regression
python src/main.py --site siteB --training-mode single --tasks classification regression
```

### Merged Dataset Training

```bash
python src/main.py --training-mode merged --tasks classification regression
```

### Custom Optimizations

```bash
# Enable specific optimizations
python src/main.py --site siteB --tasks classification --use-cascade --use-ensemble

# Disable specific optimizations
python src/main.py --site siteB --tasks classification --no-cascade --no-ensemble

# Custom feature selection
python src/main.py --site siteB --tasks classification --feature-selection --n-features 50

# With sampling methods (classification only)
python src/main.py --site siteB --tasks classification --sampler smote
python src/main.py --site siteB --tasks classification --sampler borderline
```

## Available Options

```
--site                   Site to train on (siteA, siteB, siteC)
--training-mode          single, site-specific, merged, all
--tasks                  classification, regression, or both
--sampler                none, smote, adasyn, borderline, svmsmote, kmeans, smotetomek
--use-cascade            Enable cascade classifier (default: True) [Classification only]
--no-cascade             Disable cascade classifier
--use-ensemble           Enable ensemble voting (default: True) [Both tasks]
--no-ensemble            Disable ensemble voting
--use-advanced-weights   Enable dynamic weights (default: True) [Both tasks]
--no-advanced-weights    Disable dynamic weights
--feature-selection      Enable feature selection (default: True) [Both tasks]
--no-feature-selection   Disable feature selection
--n-features             Number of features to select (default: 80)
--data-dir               Data directory (default: data)
--output-dir             Output directory (default: results)
```

## Optimizations Summary

| Optimization | Classification | Regression |
|-------------|---------------|-----------|
| Ensemble Voting | ✅ Weighted by F1 | ✅ Weighted by MAE |
| Advanced Weights | ✅ Class weights | ✅ Sample weights |
| Feature Selection | ✅ | ✅ |
| Cascade Classifier | ✅ | ❌ |
| Sampling Methods | ✅ | ❌ |

## Results

Results organized by task type:

```
results/
├── classification/
│   ├── summarize/
│   │   ├── comparison_report_<site>.txt
│   │   └── cross_site_comparison.txt (all F1 scores)
│   └── metrics/
│       └── results_<site>.json
├── regression/
│   ├── summarize/
│   │   ├── comparison_report_<site>.txt
│   │   └── cross_site_comparison.txt
│   └── metrics/
│       └── results_<site>.json
└── (models saved in ../models/)
```

- `summarize/` - Text reports and comparisons
- `metrics/` - JSON files with detailed metrics
- Classification: F1 (macro, micro, weighted, per-class), Geometric Mean, Accuracy
- Regression: MAE, RMSE, CV-RMSE, NMAE
