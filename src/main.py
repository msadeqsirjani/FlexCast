"""
Main Training and Prediction Pipeline
Energy Flexibility and Demand Response Challenge

This script trains and evaluates traditional machine learning models:

1. XGBoost
2. LightGBM
3. CatBoost
4. HistGradientBoosting (with proper Classifier/Regressor usage)

Usage:
  # Single site training
  python main.py                                              # Train on siteA (default)
  python main.py --site siteB                                 # Train on siteB

  # Site-specific training (separate model for each site)
  python main.py --training-mode site-specific                # Train siteA, siteB, siteC models

  # Merged training (single model on all sites)
  python main.py --training-mode merged                       # Train one model on all sites

  # Comprehensive training (both site-specific and merged)
  python main.py --training-mode all                          # Train both strategies

  # Advanced options
  python main.py --sampler smote                              # Use SMOTE for class balancing
  python main.py --tasks classification                       # Only classification
  python main.py --tasks regression                           # Only regression
  python main.py --tasks classification regression            # Both (default)
  python main.py --data-dir /path/to/data                     # Custom data directory
  python main.py --output-dir /path/to/results                # Custom output directory
"""

import argparse
import sys
import warnings
import logging
from pathlib import Path
from typing import List, Optional, Union

warnings.filterwarnings("ignore")

# Configure logging (WARNING level to reduce verbosity)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_VERSION = "v0.2"

# Add src to path to allow relative imports when executed from project root
if str(Path(__file__).parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from evaluation import Evaluator
from utils import save_results_json, create_summary_report
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.histgb_model import HistGradientBoostingModel
from cascade_classifier import CascadeClassifier

# For handling class imbalance
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


class FlexTrackPipeline:
    """Complete pipeline for FlexTrack Challenge"""

    def __init__(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        output_dir: Union[str, Path] = DEFAULT_RESULTS_DIR,
        sampler: str = "none",
        tasks: Optional[List[str]] = None,
        use_advanced_weights: bool = True,
        use_ensemble: bool = True,
        feature_selection: bool = True,
        n_features: int = 80,
    ):
        """
        Initialize pipeline

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for outputs
            sampler: Sampling method for class imbalance
                     ('none', 'smote', 'adasyn', 'smoteenn', 'borderline', 'svmsmote', 'kmeans', 'smotetomek')
            tasks: List of tasks to run ('classification', 'regression')
            use_advanced_weights: Use dynamic class weights (effective_samples strategy)
            use_ensemble: Add ensemble voting of top models
            feature_selection: Enable feature selection
            n_features: Number of top features to select
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sampler = sampler
        self.use_advanced_weights = use_advanced_weights
        self.use_ensemble = use_ensemble
        self.feature_selection = feature_selection
        self.n_features = n_features

        if tasks is None:
            tasks = ["classification", "regression"]
        self.tasks = list(dict.fromkeys(tasks))

        allowed_tasks = {"classification", "regression"}
        unknown_tasks = set(self.tasks) - allowed_tasks
        if unknown_tasks:
            raise ValueError(
                f"Unknown task(s) specified: {', '.join(sorted(unknown_tasks))}"
            )

        if not self.tasks:
            raise ValueError("At least one task must be specified")

        self.loader = DataLoader(data_dir)
        self.engineer = FeatureEngineer()
        self.evaluator = Evaluator()

        self.models = {}
        self.results = {}
        self.selected_features = None

    def load_and_prepare_data(self, site: str = "siteA", version: str = "v0.2"):
        """
        Load and prepare data for training

        Args:
            site: Site to train on
            version: Data version

        Returns:
            Tuple of (train_features, train_labels_class, train_labels_reg,
                     val_features, val_labels_class, val_labels_reg)
        """
        logger.info(f"Loading and preparing data for {site.upper()}")

        # Load training data
        train_data = self.loader.load_training_data(version=version)

        # Filter for specific site
        site_data = self.loader.get_site_data(site, train_data)

        # Create features
        site_data_features = self.engineer.create_all_features(
            site_data, include_lags=True, include_rolling=True
        )

        # Split into train and validation
        train_df, val_df = self.loader.split_train_validation(
            site_data_features, validation_size=0.2, time_based=True
        )

        # Get feature names
        feature_names = self.engineer.get_feature_names()

        # Prepare features and labels
        X_train = train_df[feature_names]
        y_train_class = train_df["Demand_Response_Flag"]
        y_train_reg = train_df["Demand_Response_Capacity_kW"]

        X_val = val_df[feature_names]
        y_val_class = val_df["Demand_Response_Flag"]
        y_val_reg = val_df["Demand_Response_Capacity_kW"]

        # Calculate building power stats for evaluation
        building_power = site_data["Building_Power_kW"]
        self.building_power_stats = self.evaluator.calculate_building_power_stats(
            building_power
        )

        logger.info(f"Data preparation complete: {len(feature_names)} features, {len(X_train)} train samples, {len(X_val)} val samples")

        return (
            X_train,
            y_train_class,
            y_train_reg,
            X_val,
            y_val_class,
            y_val_reg,
            feature_names,
        )

    def select_top_features(self, X_train, y_train, n_features=None):
        """
        Select top features using LightGBM feature importance

        Args:
            X_train: Training features
            y_train: Training labels
            n_features: Number of features to select

        Returns:
            List of selected feature names
        """
        if n_features is None:
            n_features = self.n_features

        logger.info(f"Selecting top {n_features} features from {len(X_train.columns)} using LightGBM importance...")

        # Train a quick LightGBM model for feature importance
        from sklearn.ensemble import RandomForestClassifier
        y_adjusted = y_train + 1

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        clf.fit(X_train, y_adjusted)

        # Get feature importances
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected = importances.head(n_features)['feature'].tolist()
        self.selected_features = selected

        logger.info(f"Selected {len(selected)} features")
        logger.info(f"Top 10: {selected[:10]}")

        return selected

    def compute_dynamic_class_weights(self, y_train, extreme_penalty: bool = False):
        """
        Compute dynamic class weights using effective samples strategy
        Better for extreme class imbalance than standard balanced weights

        Args:
            y_train: Training labels
            extreme_penalty: If True, apply 5x penalty to minority classes (disabled by default)

        Returns:
            Dictionary mapping class labels to weights
        """
        y_adjusted = y_train + 1  # Convert to 0, 1, 2
        unique_classes, counts = np.unique(y_adjusted, return_counts=True)

        # Effective number of samples strategy (better for extreme imbalance)
        # From "Class-Balanced Loss Based on Effective Number of Samples" paper
        beta = 0.9999  # Balanced beta for reasonable weights
        effective_num = 1.0 - np.power(beta, counts)
        weights_array = (1.0 - beta) / effective_num

        # Apply moderate penalty to minority classes
        if extreme_penalty:
            # Find minority classes (classes with count < 10% of max)
            max_count = counts.max()
            minority_mask = counts < (max_count * 0.1)
            weights_array[minority_mask] *= 5.0  # 5x penalty (reduced from 10x)
            logger.info("Applied 5x penalty to minority classes")

        # Create weight dictionary (convert back to -1, 0, 1 keys)
        weights = {cls - 1: w for cls, w in zip(unique_classes, weights_array)}

        logger.info(f"Dynamic class weights: {weights}")
        return weights

    def apply_resampling(self, X_train, y_train, method='smote'):
        """
        Apply advanced resampling to balance classes

        Args:
            X_train: Training features
            y_train: Training labels
            method: Resampling method
                    ('smote', 'adasyn', 'smoteenn', 'borderline', 'svmsmote', 'kmeans', 'smotetomek')

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info(f"Applying {method.upper()} for class imbalance")

        # Adjust labels for resampling (-1, 0, 1) -> (0, 1, 2)
        y_adjusted = y_train + 1

        unique, counts = np.unique(y_adjusted, return_counts=True)
        dist_str = ", ".join([f"Class {cls-1}: {count} ({count/len(y_adjusted)*100:.1f}%)" for cls, count in zip(unique, counts)])
        logger.info(f"Original distribution: {dist_str}")

        # Apply resampling based on method
        min_samples = min(counts)
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

        # For extreme imbalance, use FULL BALANCING - match majority class count
        # This is aggressive but necessary to reach F1 > 0.60
        majority_count = counts.max()
        sampling_strategy = {0: majority_count, 2: majority_count}  # Fully balance all classes

        logger.info(f"Aggressive balancing: Target {majority_count:,} samples per class")

        try:
            if method == 'smote':
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            elif method == 'adasyn':
                # ADASYN: Adaptive Synthetic Sampling
                sampler = ADASYN(random_state=42, n_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            elif method == 'smoteenn':
                # SMOTEENN: SMOTE + Edited Nearest Neighbors
                sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy))
            elif method == 'borderline':
                # BorderlineSMOTE: Focuses on borderline samples (best for imbalance)
                sampler = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, kind='borderline-1')
            elif method == 'svmsmote':
                # SVMSMOTE: Uses SVM to identify borderline samples
                sampler = SVMSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            elif method == 'kmeans':
                # KMeansSMOTE: Clusters data first, then applies SMOTE
                sampler = KMeansSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, cluster_balance_threshold=0.01)
            elif method == 'smotetomek':
                # SMOTETomek: SMOTE + Tomek links cleaning
                sampler = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy)
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            logger.debug(f"Using {method.upper()} with k_neighbors={k_neighbors}")
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_adjusted)

            # Convert back to DataFrames with original column names
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled - 1, name=y_train.name)  # Convert back to -1, 0, 1

            unique_new, counts_new = np.unique(y_resampled + 1, return_counts=True)
            dist_str_new = ", ".join([f"Class {cls-1}: {count}" for cls, count in zip(unique_new, counts_new)])
            logger.info(f"Resampled: {len(X_train)} → {len(X_resampled)} samples. Distribution: {dist_str_new}")

            return X_resampled, y_resampled

        except Exception as e:
            logger.warning(f"Resampling failed: {e}. Returning original data.")
            return X_train, y_train

    def train_all_models(
        self,
        X_train,
        y_train_class,
        y_train_reg,
        X_val,
        y_val_class,
        y_val_reg,
        tasks: Optional[List[str]] = None,
    ):
        """
        Train models for specified tasks with feature selection and ensemble

        Args:
            X_train, y_train_class, y_train_reg: Training data
            X_val, y_val_class, y_val_reg: Validation data
            tasks: List of tasks to train for
        """
        logger.info("Starting model training")

        # Feature selection for classification
        if "classification" in (tasks or self.tasks) and self.feature_selection and len(X_train.columns) > self.n_features:
            logger.info(f"Performing feature selection: {len(X_train.columns)} -> {self.n_features}")
            selected_features = self.select_top_features(X_train, y_train_class, self.n_features)
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            logger.info(f"Training with {len(selected_features)} selected features")
        else:
            logger.info(f"Training with all {len(X_train.columns)} features")

        # Traditional ML models
        model_classes = {
            "XGBoost": XGBoostModel,
            "LightGBM": LightGBMModel,
            "CatBoost": CatBoostModel,
            "HistGradientBoosting": HistGradientBoostingModel,
        }

        tasks = tasks or self.tasks

        for task in tasks:
            logger.info(f"Training {task.upper()} models")

            y_train = y_train_class if task == "classification" else y_train_reg
            y_val = y_val_class if task == "classification" else y_val_reg

            # Apply resampling for classification if enabled
            if task == "classification" and self.sampler != "none":
                X_train_balanced, y_train_balanced = self.apply_resampling(X_train, y_train, method=self.sampler)
            else:
                X_train_balanced, y_train_balanced = X_train, y_train

            # Compute dynamic class weights for advanced imbalance handling
            if task == "classification" and self.use_advanced_weights:
                class_weights = self.compute_dynamic_class_weights(y_train_balanced)
                # Convert to format for models (0, 1, 2 -> weights)
                weight_dict_adjusted = {k + 1: v for k, v in class_weights.items()}
            else:
                weight_dict_adjusted = None

            for model_name, ModelClass in model_classes.items():
                print(f"Training {model_name} ({task})...")

                try:
                    # Create model with appropriate parameters
                    # Apply dynamic class weights if available
                    if task == "classification" and weight_dict_adjusted is not None:
                        if model_name == "LightGBM":
                            # Update LightGBM params with dynamic weights
                            model = ModelClass(task=task)
                            if hasattr(model, 'params'):
                                model.params['class_weight'] = weight_dict_adjusted
                        else:
                            model = ModelClass(task=task)
                    else:
                        model = ModelClass(task=task)

                    model.train(X_train_balanced, y_train_balanced, X_val, y_val)

                    # Store model
                    key = f"{model_name}_{task}"
                    self.models[key] = model

                    # Evaluate
                    if task == "classification":
                        # Use threshold optimization for better F1 scores

                        # Get predictions
                        train_pred = model.predict(X_train)
                        val_pred = model.predict(X_val)

                        # Evaluate with predictions
                        train_results = self.evaluator.evaluate_classification(
                            y_train, train_pred, detailed=False
                        )
                        val_results = self.evaluator.evaluate_classification(
                            y_val, val_pred, detailed=True
                        )

                        print(f"  {model_name}: Train F1={train_results['f1_score_macro']:.4f}, Val F1={val_results['f1_score_macro']:.4f}, Val GM={val_results['geometric_mean_score']:.4f}")
                    else:
                        # Regression: use regular predictions
                        train_pred = model.predict(X_train)
                        val_pred = model.predict(X_val)

                        train_results = self.evaluator.evaluate_regression(
                            y_train, train_pred, self.building_power_stats
                        )
                        val_results = self.evaluator.evaluate_regression(
                            y_val, val_pred, self.building_power_stats
                        )

                        cv_rmse_str = f", CV-RMSE={val_results['cv_rmse']:.4f}" if 'cv_rmse' in val_results else ""
                        print(f"  {model_name}: Train MAE={train_results['mae']:.4f}, Val MAE={val_results['mae']:.4f}, Val RMSE={val_results['rmse']:.4f}{cv_rmse_str}")

                    # Store results
                    self.results[key] = {
                        "train": train_results,
                        "validation": val_results,
                    }

                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Create Ensemble if enabled (only for classification)
            if task == "classification" and self.use_ensemble:
                print("\nCreating Ensemble Model (Weighted Voting)...")

                try:
                    # Get probabilities from all trained models
                    model_probas = []
                    model_f1s = []

                    for model_name in ["XGBoost", "LightGBM", "CatBoost"]:
                        key = f"{model_name}_{task}"
                        if key in self.models and key in self.results:
                            model = self.models[key]
                            val_proba = model.predict_proba(X_val)
                            model_probas.append(val_proba)

                            # Get F1 score for weighting
                            f1 = self.results[key]["validation"]["f1_score_macro"]
                            model_f1s.append(f1)

                    if len(model_probas) >= 2:
                        # Weighted average based on F1 scores
                        weights = np.array(model_f1s) / sum(model_f1s)
                        ensemble_proba = np.average(model_probas, axis=0, weights=weights)

                        # Get predictions from ensemble
                        val_pred = np.argmax(ensemble_proba, axis=1) - 1  # Convert 0,1,2 back to -1,0,1

                        # Evaluate ensemble
                        val_results = self.evaluator.evaluate_classification(
                            y_val, val_pred, detailed=True
                        )

                        # Create ensemble wrapper
                        class EnsembleModel:
                            def __init__(self, models, weights):
                                self.models = models
                                self.weights = weights

                            def predict_proba(self, X):
                                probas = [m.predict_proba(X) for m in self.models]
                                return np.average(probas, axis=0, weights=self.weights)

                            def predict(self, X):
                                proba = self.predict_proba(X)
                                return np.argmax(proba, axis=1) - 1  # Convert 0,1,2 to -1,0,1

                            def save_model(self, filepath):
                                """Save ensemble - note: saves metadata only, base models saved separately"""
                                import joblib
                                joblib.dump({
                                    'weights': self.weights,
                                    'model_names': ['XGBoost', 'LightGBM', 'CatBoost']
                                }, filepath)
                                logger.debug(f"Ensemble metadata saved to {filepath}")

                        ensemble_models = [self.models[f"{name}_{task}"] for name in ["XGBoost", "LightGBM", "CatBoost"] if f"{name}_{task}" in self.models]
                        ensemble = EnsembleModel(ensemble_models, weights)

                        key = "Ensemble_classification"
                        self.models[key] = ensemble
                        self.results[key] = {
                            "train": {},
                            "validation": val_results,
                        }

                        print(f"  Ensemble: Val F1={val_results['f1_score_macro']:.4f}, Val GM={val_results['geometric_mean_score']:.4f}")
                        print(f"  Weights: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f}")

                except Exception as e:
                    logger.error(f"Error creating ensemble: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Add Cascade Classifier (two-stage hierarchical)
            if task == "classification":
                print("\nTraining CASCADE Classifier (Hierarchical)...")

                try:
                    # Create cascade with best performing models
                    cascade = CascadeClassifier(
                        stage1_model="xgboost",  # Best for Stage 1
                        stage2_model="lightgbm",  # Best for minority detection
                    )

                    # Train cascade
                    cascade.train(X_train_balanced, y_train_balanced, X_val, y_val)

                    # Get predictions
                    val_pred = cascade.predict(X_val)

                    # Evaluate
                    val_results = self.evaluator.evaluate_classification(
                        y_val, val_pred, detailed=True
                    )

                    key = "Cascade_classification"
                    self.models[key] = cascade
                    self.results[key] = {
                        "train": {},
                        "validation": val_results,
                    }

                    print(f"  Cascade: Val F1={val_results['f1_score_macro']:.4f}, Val GM={val_results['geometric_mean_score']:.4f}")

                except Exception as e:
                    logger.error(f"Error training cascade: {str(e)}")
                    import traceback
                    traceback.print_exc()

    def evaluate_and_compare(self):
        """Evaluate and compare all models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)

        if "classification" in self.tasks:
            class_results = {
                name.replace("_classification", ""): results["validation"]
                for name, results in self.results.items()
                if "classification" in name
            }

            if class_results:
                comparison_df = self.evaluator.compare_models(
                    class_results, task="classification"
                )
                print("\nCLASSIFICATION MODELS:")
                print(comparison_df.to_string(index=False))
            else:
                logger.warning("No classification models trained.")

        if "regression" in self.tasks:
            reg_results = {
                name.replace("_regression", ""): results["validation"]
                for name, results in self.results.items()
                if "regression" in name
            }

            if reg_results:
                comparison_df = self.evaluator.compare_models(
                    reg_results, task="regression"
                )
                print("\nREGRESSION MODELS:")
                print(comparison_df.to_string(index=False))
            else:
                logger.warning("No regression models trained.")

    def save_results(self, site: str = "siteA", subfolder: str = None):
        """Save all results"""
        print("\nSaving results...")

        # Save results JSON
        results_path = self.output_dir / f"results_{site}.json"
        save_results_json(self.results, str(results_path))

        # Create summary report
        report_path = self.output_dir / f"comparison_report_{site}.txt"
        all_results = {
            name: results["validation"] for name, results in self.results.items()
        }
        create_summary_report(
            all_results,
            str(report_path),
            task=self._get_summary_task_mode(),
        )

        # Save models
        if subfolder:
            models_dir = Path("../models") / subfolder
        else:
            models_dir = Path("../models")
        models_dir.mkdir(exist_ok=True, parents=True)

        for name, model in self.models.items():
            # Skip saving optimizer objects (they're stored alongside models)
            if "_optimizer" not in name:
                model_path = models_dir / f"{name}_{site}.pkl"
                model.save_model(str(model_path))

        print(f"  Results: {results_path}")
        print(f"  Models: {models_dir}")

    def _get_summary_task_mode(self) -> str:
        if "classification" in self.tasks and "regression" in self.tasks:
            return "both"
        if "classification" in self.tasks:
            return "classification"
        return "regression"

    def prepare_merged_data(self, sites: list, version: str = "v0.2"):
        """Prepare merged data from multiple sites"""
        logger.info(f"Preparing merged data from: {', '.join(sites)}")

        train_data = self.loader.load_training_data(version=version)
        merged_data = train_data[train_data["Site"].isin(sites)].copy()

        site_counts = ", ".join([f"{site}: {len(merged_data[merged_data['Site'] == site])}" for site in sites])
        logger.info(f"Total samples: {len(merged_data)} ({site_counts})")

        # Create features
        merged_data_features = self.engineer.create_all_features(
            merged_data, include_lags=True, include_rolling=True
        )

        # Split
        train_df, val_df = self.loader.split_train_validation(
            merged_data_features, validation_size=0.2, time_based=True
        )

        feature_names = self.engineer.get_feature_names()
        X_train, y_train_class, y_train_reg = (
            train_df[feature_names],
            train_df["Demand_Response_Flag"],
            train_df["Demand_Response_Capacity_kW"],
        )
        X_val, y_val_class, y_val_reg = (
            val_df[feature_names],
            val_df["Demand_Response_Flag"],
            val_df["Demand_Response_Capacity_kW"],
        )

        self.building_power_stats = self.evaluator.calculate_building_power_stats(
            merged_data["Building_Power_kW"]
        )

        logger.info(f"Features: {len(feature_names)}, Train: {len(X_train)}, Val: {len(X_val)}")
        return (
            X_train,
            y_train_class,
            y_train_reg,
            X_val,
            y_val_class,
            y_val_reg,
            feature_names,
        )

    def run_site_specific_training(
        self,
        sites: list = ["siteA", "siteB", "siteC"],
        version: str = "v0.2",
    ):
        """Train separate models for each site"""
        logger.info(f"Site-specific training: {', '.join(sites)}")

        all_site_results = {}
        for site in sites:
            logger.info(f"{'='*40} SITE: {site.upper()} {'='*40}")
            X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names = (
                self.load_and_prepare_data(site=site, version=version)
            )
            self.feature_names = feature_names

            self.models, self.results = {}, {}
            self.train_all_models(
                X_train,
                y_train_class,
                y_train_reg,
                X_val,
                y_val_class,
                y_val_reg,
                tasks=self.tasks,
            )

            all_site_results[site] = {
                "models": self.models.copy(),
                "results": self.results.copy(),
            }
            self.save_results(site=site, subfolder=f"site_specific/{site}")

        self._print_cross_site_comparison(all_site_results)
        return all_site_results

    def run_merged_training(
        self,
        sites: list = ["siteA", "siteB", "siteC"],
        version: str = "v0.2",
    ):
        """Train single merged model on all sites"""
        logger.info(f"Merged model training: {', '.join(sites)}")

        X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names = (
            self.prepare_merged_data(sites=sites, version=version)
        )
        self.feature_names = feature_names

        self.models, self.results = {}, {}
        self.train_all_models(
            X_train,
            y_train_class,
            y_train_reg,
            X_val,
            y_val_class,
            y_val_reg,
            tasks=self.tasks,
        )
        self.evaluate_and_compare()
        self.save_results(site="merged", subfolder="merged")

        return self.models, self.results

    def _print_cross_site_comparison(self, all_site_results: dict):
        """Print cross-site comparison"""
        logger.info("Cross-site performance comparison")

        if "classification" in self.tasks:
            lines = ["\n--- CLASSIFICATION ---", f"{'Model':<20} {'Site':<10} {'F1':>10} {'GM':>10}", "-" * 60]
            for site, data in all_site_results.items():
                for name, res in data["results"].items():
                    if "classification" in name:
                        vr = res["validation"]
                        lines.append(f"{name.replace('_classification', ''):<20} {site:<10} {vr['f1_score_macro']:>10.4f} {vr['geometric_mean_score']:>10.4f}")
            logger.info("\n".join(lines))

        if "regression" in self.tasks:
            lines = ["\n--- REGRESSION ---", f"{'Model':<20} {'Site':<10} {'MAE':>10} {'RMSE':>10} {'CV-RMSE':>10}", "-" * 70]
            for site, data in all_site_results.items():
                for name, res in data["results"].items():
                    if "regression" in name:
                        vr = res["validation"]
                        cv_rmse_val = vr.get('cv_rmse', 0.0)
                        lines.append(f"{name.replace('_regression', ''):<20} {site:<10} {vr['mae']:>10.4f} {vr['rmse']:>10.4f} {cv_rmse_val:>10.4f}")
            logger.info("\n".join(lines))

        # Save to file
        comp_path = self.output_dir / "cross_site_comparison.txt"
        with open(comp_path, "w") as f:
            f.write("CROSS-SITE PERFORMANCE COMPARISON\n" + "=" * 80 + "\n")
            for site, data in all_site_results.items():
                f.write(f"\n{site.upper()}:\n")
                for name, res in data["results"].items():
                    vr = res["validation"]
                    f.write(f"  {name}:\n")
                    if "classification" in name:
                        f.write(
                            f"    F1: {vr['f1_score_macro']:.4f}, GM: {vr['geometric_mean_score']:.4f}\n"
                        )
                    elif "regression" in name:
                        cv_rmse_val = vr.get('cv_rmse', 0.0)
                        f.write(
                            f"    MAE: {vr['mae']:.4f}, RMSE: {vr['rmse']:.4f}, CV-RMSE: {cv_rmse_val:.4f}\n"
                        )
        logger.info(f"Cross-site comparison saved to: {comp_path}")

    def run_full_pipeline(
        self,
        site: str = "siteA",
        version: str = "v0.2",
        training_mode: str = "single",
    ):
        """
        Run the complete pipeline

        Args:
            site: Site to train on (for single mode)
            version: Data version
            training_mode: 'single', 'site-specific', 'merged', or 'all'
        """
        logger.info(f"FlexTrack Challenge Pipeline - Mode: {training_mode}, Tasks: {', '.join(self.tasks)}")

        sites = ["siteA", "siteB", "siteC"]

        if training_mode == "single":
            X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names = (
                self.load_and_prepare_data(site=site, version=version)
            )
            self.feature_names = feature_names
            self.train_all_models(
                X_train,
                y_train_class,
                y_train_reg,
                X_val,
                y_val_class,
                y_val_reg,
            )
            self.evaluate_and_compare()
            self.save_results(site=site)

        elif training_mode == "site-specific":
            self.run_site_specific_training(sites=sites, version=version)

        elif training_mode == "merged":
            self.run_merged_training(sites=sites, version=version)

        elif training_mode == "all":
            logger.info("Starting comprehensive training (site-specific + merged)")
            self.run_site_specific_training(sites=sites, version=version)
            self.run_merged_training(sites=sites, version=version)
            logger.info(f"✓ Trained {len(sites)} site-specific + 1 merged model")
            logger.info("✓ Results: results/, Models: models/")

        logger.info("Pipeline completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FlexTrack Challenge - Energy Flexibility Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--site", type=str, default="siteA", help="Site to train on (default: siteA)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.2",
        help="Data version (default: v0.2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Data directory (default: ../data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Output directory (default: ../results)",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="single",
        choices=["single", "site-specific", "merged", "all"],
        help="Training mode: single site, site-specific, merged, or all (default: single)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="none",
        choices=["none", "smote", "adasyn", "smoteenn", "borderline", "svmsmote", "kmeans", "smotetomek"],
        help="Sampling method for class imbalance (default: none). Advanced options: borderline (recommended), svmsmote, kmeans, smotetomek",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["classification", "regression"],
        default=["classification", "regression"],
        help="Tasks to run: specify one or both of classification and regression (default: both)",
    )
    parser.add_argument(
        "--use-advanced-weights",
        action="store_true",
        default=True,
        help="Use dynamic class weights based on effective samples (default: True, use --no-advanced-weights to disable)",
    )
    parser.add_argument(
        "--no-advanced-weights",
        dest="use_advanced_weights",
        action="store_false",
        help="Disable advanced class weighting",
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        default=True,
        help="Enable ensemble voting (default: True, use --no-ensemble to disable)",
    )
    parser.add_argument(
        "--no-ensemble",
        dest="use_ensemble",
        action="store_false",
        help="Disable ensemble voting",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        default=True,
        help="Enable feature selection (default: True, use --no-feature-selection to disable)",
    )
    parser.add_argument(
        "--no-feature-selection",
        dest="feature_selection",
        action="store_false",
        help="Disable feature selection",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=80,
        help="Number of features to select (default: 80)",
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = FlexTrackPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampler=args.sampler,
        tasks=args.tasks,
        use_advanced_weights=args.use_advanced_weights,
        use_ensemble=args.use_ensemble,
        feature_selection=args.feature_selection,
        n_features=args.n_features,
    )

    pipeline.run_full_pipeline(
        site=args.site,
        version=args.version,
        training_mode=args.training_mode,
    )


if __name__ == "__main__":
    main()
