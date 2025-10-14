# Energy Flexibility and Demand Response Challenge

## Problem Statement
As modern power grids integrate renewable energy sources, maintaining balance between supply and demand becomes increasingly difficult. This project applies AI methods to:
1. **Detect demand response events** (Classification: -1, 0, +1)
2. **Predict energy flexibility** (Regression: kW capacity shifted)

## Project Structure
```
energy-flexibility-dr-project/
├── data/                  # Training and test datasets
├── src/                   # Source code modules
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── models/           # Model implementations
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   └── histgb_model.py
│   ├── evaluation.py     # Evaluation metrics
│   └── utils.py          # Utility functions
├── models/               # Saved trained models
├── results/              # Predictions and reports
├── notebooks/            # Jupyter notebooks for exploration
├── logs/                 # Training logs
└── requirements.txt      # Dependencies
```

## Methods Implemented
1. **XGBoost** - Extreme Gradient Boosting with strong regularization
2. **LightGBM** - Fast, histogram-based gradient boosting
3. **CatBoost** - Handles categorical features natively
4. **HistGradientBoosting** - Scikit-learn's native implementation

## Metrics
- **Classification**: F1 Score, Geometric Mean Score
- **Regression**: MAE, RMSE, Normalized MAE, Normalized RMSE

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train all models
python src/main.py --mode train

# Make predictions
python src/main.py --mode predict

# Compare models
python src/main.py --mode compare
```

## Dataset
FlexTrack Challenge 2025 dataset with 15-min resolution:
- Building power consumption
- Outdoor temperature
- Solar radiation
- DR event flag (-1, 0, +1)
- DR capacity (kW)

## Author
Data Science Project - FlexTrack Challenge 2025
