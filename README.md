# FlexCast - Energy Flexibility Forecasting

## Installation

```bash
git clone <repository-url>
cd FlexCast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

### All Optimizations on All Sites (Classification)

```bash
python src/main.py --training-mode site-specific --tasks classification --sampler none --use-advanced-weights --use-ensemble --use-cascade --feature-selection --n-features 80 --data-dir data --output-dir results
```

### Single Site Training

```bash
python src/main.py --site siteB --training-mode single --tasks classification
```

### All Sites Training

```bash
python src/main.py --training-mode site-specific --tasks classification
```

### Merged Dataset Training

```bash
python src/main.py --training-mode merged --tasks classification
```

### With Specific Optimizations

```bash
python src/main.py --site siteB --tasks classification --use-cascade --use-ensemble
python src/main.py --site siteB --tasks classification --no-cascade --no-ensemble
python src/main.py --site siteB --tasks classification --feature-selection --n-features 50
```

### Classification and Regression

```bash
python src/main.py --site siteB --tasks classification regression
python src/main.py --site siteB --tasks regression
```

### With Sampling Methods

```bash
python src/main.py --site siteB --tasks classification --sampler smote
python src/main.py --site siteB --tasks classification --sampler borderline
python src/main.py --site siteB --tasks classification --sampler adasyn
```

## Available Options

```
--site                   Site to train on (siteA, siteB, siteC)
--training-mode          single, site-specific, merged, all
--tasks                  classification, regression, or both
--sampler                none, smote, adasyn, borderline, svmsmote, kmeans, smotetomek
--use-cascade            Enable cascade classifier (default: True)
--no-cascade             Disable cascade classifier
--use-ensemble           Enable ensemble voting (default: True)
--no-ensemble            Disable ensemble voting
--use-advanced-weights   Enable dynamic class weights (default: True)
--no-advanced-weights    Disable dynamic class weights
--feature-selection      Enable feature selection (default: True)
--no-feature-selection   Disable feature selection
--n-features             Number of features to select (default: 80)
--data-dir               Data directory (default: data)
--output-dir             Output directory (default: results)
```

## Results

Results saved to: `results/<experiment>/`
- `results_<site>.json` - Detailed metrics
- `comparison_report_<site>.txt` - Summary report
- `models/` - Trained model files
