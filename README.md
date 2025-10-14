# FlexCast - Energy Flexibility Forecasting

**Advanced Machine Learning & Deep Learning for Energy Flexibility and Demand Response Prediction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FlexTrack Challenge 2025](https://img.shields.io/badge/challenge-FlexTrack%202025-orange.svg)](https://www.aicrowd.com/challenges/flextrack-challenge-2025)

> A comprehensive ML/DL pipeline for predicting demand response events and capacity in commercial buildings. Supports traditional ML models (XGBoost, LightGBM, CatBoost, HistGradientBoosting) and deep learning architectures (LSTM, GRU, CNN, TCN, Transformer).

---

## 🎯 Overview

FlexCast predicts energy flexibility in commercial buildings using state-of-the-art machine learning and deep learning models:

1. **Demand Response Classification**: Predict whether a building will participate in demand response events
2. **Demand Response Capacity Regression**: Predict how much power the building can reduce (in kW)

### Key Features

✨ **9 Model Architectures**
- Traditional ML: XGBoost, LightGBM, CatBoost, HistGradientBoosting
- Deep Learning: LSTM, GRU, CNN, TCN, Transformer

🏢 **Multi-Site Training**
- Site-specific models (one per building site)
- Merged models (trained on all sites combined)
- Comprehensive cross-site evaluation

🔧 **Advanced Feature Engineering**
- Temporal features with cyclical encoding
- Lag features (1, 2, 4, 8, 12, 24, 48, 96 time steps)
- Rolling statistics (mean, std, min, max)
- Interaction features

📊 **Comprehensive Evaluation**
- Classification: F1 Score, Geometric Mean Score
- Regression: MAE, RMSE, Normalized MAE/RMSE, CV-RMSE

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

# For deep learning models (optional)
pip install torch torchvision
```

### Run the Pipeline

```bash
cd src

# Train traditional ML models on single site
python main.py

# Train all models (ML + DL) on all sites
python main.py --training-mode all --models-type all

# Train site-specific models
python main.py --training-mode site-specific

# Train merged model on all sites
python main.py --training-mode merged
```

---

## 📁 Project Structure

```
energy-flexibility-dr-project/
├── data/                           # Training and test data (*.csv)
├── src/                            # Source code
│   ├── main.py                     # Main pipeline (ML + DL + multi-site)
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── feature_engineering.py      # Feature creation
│   ├── evaluation.py               # Metrics and evaluation
│   ├── utils.py                    # Utility functions
│   └── models/                     # Model implementations
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       ├── catboost_model.py
│       ├── histgb_model.py
│       ├── base_dl_model.py        # Base class for deep learning
│       ├── lstm_model.py
│       ├── gru_model.py
│       ├── cnn_model.py
│       ├── tcn_model.py
│       └── transformer_model.py
├── models/                         # Saved trained models
│   ├── site_specific/              # Site-specific models
│   │   ├── siteA/
│   │   ├── siteB/
│   │   └── siteC/
│   └── merged/                     # Merged models
├── results/                        # Results and reports
├── notebooks/                      # Jupyter notebooks
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 💻 Usage

### Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --site SITE                    Site to train on (siteA, siteB, siteC) [default: siteA]
  --version VERSION              Data version (v0.1, v0.2) [default: v0.2]
  --data-dir PATH                Data directory [default: ../data]
  --output-dir PATH              Output directory [default: ../results]
  --models-type TYPE             Model type: traditional, deep-learning, all [default: traditional]
  --training-mode MODE           Training mode: single, site-specific, merged, all [default: single]
  --sequence-length N            Sequence length for DL models [default: 96]
  --dl-epochs N                  Training epochs for DL [default: 50]
  --dl-batch-size N              Batch size for DL [default: 64]
```

### Training Modes

#### 1. Single Site Training
Train models on one site only:
```bash
python main.py --site siteA
python main.py --site siteB --models-type all
```

#### 2. Site-Specific Training
Train separate models for each site (siteA, siteB, siteC):
```bash
python main.py --training-mode site-specific
python main.py --training-mode site-specific --models-type all
```

Output:
- 3 separate models (one per site)
- Cross-site performance comparison

#### 3. Merged Training
Train a single model on data from all sites:
```bash
python main.py --training-mode merged
python main.py --training-mode merged --models-type deep-learning
```

Output:
- 1 model trained on combined data

#### 4. Comprehensive Training
Train both site-specific and merged models:
```bash
python main.py --training-mode all
python main.py --training-mode all --models-type all
```

Output:
- 3 site-specific models
- 1 merged model
- Comprehensive comparison

### Model Types

```bash
# Traditional ML only (XGBoost, LightGBM, CatBoost, HistGradientBoosting)
python main.py --models-type traditional

# Deep Learning only (LSTM, GRU, CNN, TCN, Transformer)
python main.py --models-type deep-learning

# All models
python main.py --models-type all
```

### Deep Learning Options

```bash
# Customize sequence length (time steps)
python main.py --models-type deep-learning --sequence-length 48

# Adjust training epochs
python main.py --models-type deep-learning --dl-epochs 100

# Change batch size
python main.py --models-type deep-learning --dl-batch-size 32

# Complete example
python main.py --training-mode all --models-type deep-learning \
    --sequence-length 96 --dl-epochs 100 --dl-batch-size 32
```

---

## 🤖 Models

### Traditional Machine Learning

| Model | Description | Best For |
|-------|-------------|----------|
| **XGBoost** | Extreme gradient boosting | High accuracy, robust |
| **LightGBM** | Fast gradient boosting | Large datasets, speed |
| **CatBoost** | Handles categorical features | Mixed data types |
| **HistGradientBoosting** | Histogram-based boosting | Memory efficiency |

### Deep Learning

| Model | Architecture | Best For |
|-------|-------------|----------|
| **LSTM** | Long Short-Term Memory | Long-term dependencies |
| **GRU** | Gated Recurrent Unit | Faster than LSTM, good performance |
| **CNN** | 1D Convolutional Network | Local patterns, efficient |
| **TCN** | Temporal Convolutional Network | Large receptive field |
| **Transformer** | Self-attention mechanism | Complex temporal relationships |

---

## 📊 Evaluation Metrics

### Classification Task
- **F1 Score (Macro)**: Primary metric (harmonic mean of precision/recall)
- **Geometric Mean Score**: For imbalanced classes
- Precision, Recall, Accuracy, Confusion Matrix

### Regression Task
- **MAE** (Mean Absolute Error): Primary metric
- **RMSE** (Root Mean Squared Error)
- **Normalized MAE**: MAE / mean building power
- **Normalized RMSE**: RMSE / mean building power
- **CV-RMSE**: Coefficient of variation RMSE

---

## 📈 Output Files

```
results/
├── results_siteA.json                  # Metrics for siteA
├── results_siteB.json                  # Metrics for siteB
├── results_siteC.json                  # Metrics for siteC
├── results_merged.json                 # Metrics for merged model
├── comparison_report_siteA.txt         # Model comparison for siteA
├── comparison_report_merged.txt        # Model comparison for merged
└── cross_site_comparison.txt           # Cross-site performance

models/
├── site_specific/
│   ├── siteA/
│   │   ├── XGBoost_classification_siteA.pkl
│   │   ├── XGBoost_regression_siteA.pkl
│   │   ├── LSTM_classification_siteA.pth
│   │   └── ...
│   ├── siteB/
│   └── siteC/
└── merged/
    ├── XGBoost_classification_merged.pkl
    ├── LSTM_classification_merged.pth
    └── ...
```

---

## 🔧 Feature Engineering

### Temporal Features
- Hour of day (0-23) with sin/cos encoding
- Day of week (0-6) with sin/cos encoding
- Month (1-12) with sin/cos encoding
- Is weekend flag

### Lag Features
Previous values at time steps: 1, 2, 4, 8, 12, 24, 48, 96
- Building power lags
- Temperature lags
- Solar irradiance lags

### Rolling Statistics
Windows: 4, 8, 12, 24, 48, 96 time steps
- Mean, standard deviation, min, max
- Applied to building power and weather features

### Interaction Features
- Temperature × Solar Irradiance
- Building Power × Temperature
- Building Power × Hour
- Custom domain-specific interactions

---

## 📉 Example Results

### Classification Performance
```
Model                    F1 Score    GM Score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost                  0.8542      0.8631
LightGBM                 0.8501      0.8598
LSTM                     0.8423      0.8512
Transformer              0.8389      0.8478
```

### Regression Performance
```
Model                    MAE         RMSE        CV-RMSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost                  12.45       18.23       0.1523
LightGBM                 12.67       18.45       0.1541
GRU                      13.12       19.34       0.1612
TCN                      13.45       19.78       0.1648
```

### Cross-Site Comparison
```
Model            Site       F1 Score    MAE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost         siteA       0.8542     12.45
XGBoost         siteB       0.8321     14.67
XGBoost         siteC       0.8198     13.89
Merged Model    all         0.8401     13.67
```

---

## 🛠️ Troubleshooting

### Issue: PyTorch not available
```bash
# Install PyTorch
pip install torch torchvision

# Or use CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: CUDA out of memory
```bash
# Reduce batch size
python main.py --models-type deep-learning --dl-batch-size 16

# Reduce sequence length
python main.py --models-type deep-learning --sequence-length 48
```

### Issue: Training is too slow
```bash
# Train fewer models
python main.py --models-type traditional

# Reduce epochs
python main.py --models-type deep-learning --dl-epochs 20

# Use single site instead of all sites
python main.py --training-mode single --site siteA
```

### Issue: FileNotFoundError
Ensure data files are in `data/` directory with naming pattern:
- Training: `*flextrack-2025-training-data-v0.2.csv`
- Test: `*flextrack-2025-public-test-data-v0.2.csv`

---

## 🎓 Research & Development

### Model Selection Guidelines

| Scenario | Recommended Approach |
|----------|---------------------|
| **Maximum Accuracy** | XGBoost or Ensemble of top models |
| **Fast Prototyping** | LightGBM or GRU |
| **Production Deployment** | LightGBM or GRU (speed + accuracy) |
| **Research/Innovation** | Transformer or TCN |
| **Limited Compute** | Traditional ML models only |
| **Interpretability** | XGBoost + SHAP values |

### Hyperparameter Tuning

For production use, consider tuning:
- Learning rate
- Number of estimators/layers
- Regularization parameters
- Sequence length for DL models
- Hidden layer sizes

Use libraries like Optuna or Ray Tune for automated hyperparameter optimization.

---

## 📚 References

- FlexTrack Challenge 2025: https://www.aicrowd.com/challenges/flextrack-challenge-2025
- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- CatBoost: Prokhorenkova et al. (2018)
- LSTM: Hochreiter & Schmidhuber (1997)
- Transformer: Vaswani et al. (2017)
- TCN: Bai et al. (2018)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**FlexTrack Challenge 2025**
- Email: eyap@uow.edu.au
- Challenge Website: https://www.aicrowd.com/challenges/flextrack-challenge-2025

---

## 🙏 Acknowledgments

- FlexTrack Challenge 2025 organizers
- University of Wollongong
- Energy Systems Research Lab
- Open source ML/DL community

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for the FlexTrack Challenge 2025

</div>
