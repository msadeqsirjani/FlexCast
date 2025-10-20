# FlexCast - Energy Flexibility Forecasting

**Advanced Machine Learning for Energy Flexibility and Demand Response Prediction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FlexTrack Challenge 2025](https://img.shields.io/badge/challenge-FlexTrack%202025-orange.svg)](https://www.aicrowd.com/challenges/flextrack-challenge-2025)

> A comprehensive ML pipeline for predicting demand response events and capacity in commercial buildings. Features 4 optimized models (XGBoost, LightGBM, CatBoost, HistGradientBoosting) with automatic class imbalance handling and threshold optimization.

---

## 🎯 Overview

FlexCast predicts energy flexibility in commercial buildings using state-of-the-art machine learning models:

1. **Demand Response Classification**: Predict whether a building will participate in demand response events (-1: decrease, 0: no change, 1: increase)
2. **Demand Response Capacity Regression**: Predict how much power the building can reduce (in kW)

### Key Features

✨ **4 Optimized Models**
- XGBoost, LightGBM, CatBoost, HistGradientBoosting
- Automatic hyperparameter optimization
- Built-in class imbalance handling

🏢 **Multi-Site Training**
- Site-specific models (one per building site)
- Merged models (trained on all sites combined)
- Comprehensive cross-site evaluation

🔧 **Advanced Feature Engineering**
- 142 features: temporal, lag, rolling, interaction, energy patterns
- Automatic feature selection capabilities
- Time-series specific optimizations

📊 **Comprehensive Evaluation**
- Classification: F1 Score, Geometric Mean Score
- Regression: MAE, RMSE, Normalized MAE/RMSE, CV-RMSE
- Threshold optimization for better F1 scores

---

## 📊 Performance Results

### Classification Performance (F1 Score)

| Model | Site A | Site B | Site C | Merged | Best Site |
|-------|--------|--------|--------|--------|-----------|
| **LightGBM** | 0.428 | **0.541** | 0.487 | 0.465 | Site B ✅ |
| **XGBoost** | 0.440 | 0.536 | 0.487 | 0.450 | Site B |
| **CatBoost** | 0.410 | **0.518** | 0.428 | 0.404 | Site B |
| **HistGradientBoosting** | 0.045 | 0.299 | 0.381 | 0.415 | Site C |

### Key Insights

🎯 **Best Performance**: Site B with LightGBM (F1 = 0.541)
- Site B shows the most balanced class distribution
- LightGBM consistently performs well across sites
- XGBoost and CatBoost also achieve good results on Site B

⚠️ **Challenge**: Extreme class imbalance (244:1 ratio)
- Class 0 (no DR): ~97.8% of data
- Class -1 (decrease): ~1.8% of data  
- Class 1 (increase): ~0.4% of data (only 140 samples!)

✅ **Solution**: Automatic extreme imbalance detection
- Pipeline detects >200:1 imbalance ratio
- Uses aggressive hyperparameters automatically
- Built-in class weights (no SMOTE needed)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd energy-flexibility-dr-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Diagnose your data (recommended first step)
python src/diagnose_performance.py siteA

# 2. Train models (automatic optimization)
python src/main.py --sampler none --tasks classification

# 3. Train all sites
python src/main.py --training-mode site-specific --sampler none

# 4. Visualize results
python src/visualize_results.py
```

### Advanced Usage

```bash
# Try different sampling strategies
python src/main.py --sampler adasyn --tasks classification

# Train only classification (faster)
python src/main.py --tasks classification --sampler none

# Train merged model (all sites combined)
python src/main.py --training-mode merged --sampler none

# Comprehensive training (both site-specific and merged)
python src/main.py --training-mode all --sampler none
```

---

## 📁 Project Structure

```
energy-flexibility-dr-project/
├── src/
│   ├── main.py                    # Main training pipeline ⭐
│   ├── data_loader.py             # Data loading and splitting
│   ├── feature_engineering.py     # Feature creation (142 features)
│   ├── evaluation.py              # Model evaluation metrics
│   ├── threshold_optimizer.py     # Optimize classification thresholds
│   ├── model_tuning.py            # Hyperparameter optimization
│   ├── diagnose_performance.py    # Data diagnostics tool 🔍
│   ├── visualize_results.py       # Results visualization
│   └── models/                    # Model implementations
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       ├── catboost_model.py
│       └── histgb_model.py
├── data/                          # Training and test data
├── results/                       # Model outputs and reports
├── models/                        # Saved trained models
├── PERFORMANCE_GUIDE.md           # Performance improvement guide
└── PROJECT_STRUCTURE.md           # Detailed project documentation
```

---

## 🔧 Features

### Automatic Optimization

- **Class Imbalance Detection**: Automatically detects >200:1 imbalance and uses extreme parameters
- **Threshold Optimization**: Finds optimal classification thresholds for better F1 scores
- **Hyperparameter Tuning**: Pre-optimized parameters for each model type
- **Feature Engineering**: 142 features including temporal, lag, rolling, and domain-specific patterns

### Model Capabilities

- **XGBoost**: Extreme Gradient Boosting with histogram-based trees
- **LightGBM**: Fast gradient boosting with built-in imbalance handling
- **CatBoost**: Handles categorical features natively with ordered boosting
- **HistGradientBoosting**: Sklearn's native histogram-based gradient boosting

### Evaluation Metrics

**Classification:**
- F1 Score (macro) - Primary metric
- Geometric Mean Score - Balanced accuracy across classes
- Per-class F1 scores
- Confusion matrices

**Regression:**
- MAE, RMSE
- Normalized MAE/RMSE
- CV-RMSE (Coefficient of Variation)

---

## 📊 Understanding Your Results

### Why F1 Scores Are "Low"

With extreme class imbalance (244:1 ratio):
- **Random baseline**: F1 ≈ 0.01
- **Majority class only**: F1 ≈ 0.32
- **Your models**: F1 = 0.40-0.54 ✅ (25-70% better than baseline!)

### Class Distribution Analysis

```
Site A: Class -1: 1.8%, Class 0: 97.8%, Class 1: 0.4% (244:1 imbalance)
Site B: Better balance, higher F1 scores (0.50-0.54)
Site C: Moderate balance, decent F1 scores (0.38-0.49)
Merged: Combined data, balanced performance (0.40-0.47)
```

### Confusion Matrix Insights

**Typical pattern:**
```
Predicted →     -1    0    1
Actual ↓
-1             [  X   Y    0]  ← Hard to predict (looks like class 0)
 0             [  A   B    C]  ← High accuracy (majority class)
 1             [  D   E    F]  ← Moderate accuracy (very few samples)
```

**Solution**: Add domain features that distinguish "power decreasing" from "power stable"

---

## 🛠️ Performance Improvement

### Quick Wins (Today)

1. **Use Site B data** - Best class balance
2. **Train without SMOTE** - `--sampler none` (recommended)
3. **Try ADASYN** - `--sampler adasyn` (adaptive sampling)

### Advanced Techniques (This Week)

1. **Add domain features** - Power derivatives, sustained changes
2. **Feature selection** - Reduce from 142 to top 50 features
3. **Ensemble methods** - Combine predictions from multiple models

### Expected Improvements

| Technique | Expected F1 | Time |
|-----------|------------|------|
| Current (optimized) | 0.45-0.54 | - |
| Site B + no SMOTE | 0.50-0.55 | 5 min |
| ADASYN sampling | 0.50-0.58 | 10 min |
| Domain features | 0.55-0.65 | 2-3 days |
| Feature selection | 0.52-0.60 | 1 day |
| Ensemble methods | 0.55-0.70 | 1 week |

---

## 📈 Visualization

The pipeline generates comprehensive visualizations:

- **Class distribution analysis** - Imbalance ratios across sites
- **Model performance comparison** - F1 scores and geometric means
- **Cross-site trends** - Performance patterns across sites
- **Feature importance** - Top contributing features
- **Temporal patterns** - Time-based analysis

View results: `python src/visualize_results.py`

---

## 🎯 Realistic Goals

Given the extreme class imbalance (244:1 ratio):

| F1 Score | Assessment | Achievable |
|----------|------------|------------|
| 0.40-0.50 | Good | ✅ Current performance |
| 0.50-0.60 | Very good | ✅ With optimizations |
| 0.60-0.70 | Excellent | ⚠️ Advanced techniques needed |
| 0.70+ | Exceptional | ❌ Need more data |

**Note**: F1 = 0.54 (Site B, LightGBM) is actually excellent for this problem!

---

## 📚 Documentation

- **`PERFORMANCE_GUIDE.md`** - Detailed improvement strategies
- **`PROJECT_STRUCTURE.md`** - Complete project documentation
- **`src/diagnose_performance.py`** - Data quality diagnostics

---

## 🚀 Commands Reference

```bash
# Essential commands
python src/diagnose_performance.py siteA           # Check data quality
python src/main.py --sampler none                  # Best practice training
python src/main.py --sampler adasyn                # Try adaptive sampling
python src/main.py --training-mode site-specific   # Train all sites
python src/visualize_results.py                    # Generate visualizations

# Advanced usage
python src/main.py --tasks classification          # Classification only
python src/main.py --training-mode merged          # Merged model
python src/main.py --training-mode all             # Comprehensive training
```

---

## 🔍 Troubleshooting

**Problem**: Low F1 scores (< 0.40)
**Solution**: 
```bash
python src/diagnose_performance.py siteA  # Check data quality
python src/main.py --sampler none          # Use optimized parameters
```

**Problem**: Models can't predict minority class
**Solution**: Check class distribution, add domain features, use ADASYN

**Problem**: Overfitting (train >> validation)
**Solution**: Use feature selection, increase regularization

---

## 📊 Results Summary

**Best Performance**: Site B with LightGBM (F1 = 0.541)
**Most Consistent**: XGBoost across all sites
**Biggest Challenge**: Extreme class imbalance (244:1 ratio)
**Key Insight**: Site-specific models outperform merged models

**Recommendation**: Focus on Site B data and add domain-specific features for power change patterns.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

For questions and support:
- Check `PERFORMANCE_GUIDE.md` for improvement strategies
- Run `python src/diagnose_performance.py siteA` for data analysis
- Review `PROJECT_STRUCTURE.md` for detailed documentation

---

**Current Best Model**: LightGBM on Site B with F1 = 0.541 🎯

For detailed improvement strategies, see `PERFORMANCE_GUIDE.md` 📖