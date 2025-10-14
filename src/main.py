"""
Main Training and Prediction Pipeline
Energy Flexibility and Demand Response Challenge

This script trains and evaluates models:

Traditional ML Models:
1. XGBoost
2. LightGBM
3. CatBoost
4. HistGradientBoosting (with proper Classifier/Regressor usage)

Deep Learning Models (requires PyTorch):
5. LSTM (Long Short-Term Memory)
6. GRU (Gated Recurrent Unit)
7. CNN (1D Convolutional Neural Network)
8. TCN (Temporal Convolutional Network)
9. Transformer

Usage:
  # Single site training
  python main.py                                              # Train on siteA (traditional ML)
  python main.py --site siteB                                 # Train on siteB
  python main.py --models-type all                            # Train all models (ML + DL)

  # Site-specific training (separate model for each site)
  python main.py --training-mode site-specific                # Train siteA, siteB, siteC models
  python main.py --training-mode site-specific --models-type all

  # Merged training (single model on all sites)
  python main.py --training-mode merged                       # Train one model on all sites

  # Comprehensive training (both site-specific and merged)
  python main.py --training-mode all                          # Train both strategies
  python main.py --training-mode all --models-type all        # All strategies + all models
"""

import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from evaluation import Evaluator
from utils import save_results_json, create_summary_report
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.histgb_model import HistGradientBoostingModel

# Try to import deep learning models (requires PyTorch)
DEEP_LEARNING_AVAILABLE = False
try:
    from models.lstm_model import LSTMModel
    from models.gru_model import GRUModel
    from models.cnn_model import CNNModel
    from models.tcn_model import TCNModel
    from models.transformer_model import TransformerModel

    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Deep learning models not available. Install PyTorch to use them.")
    print(f"Error: {e}")


class FlexTrackPipeline:
    """Complete pipeline for FlexTrack Challenge"""

    def __init__(
        self,
        data_dir: str = "../data",
        output_dir: str = "../results",
        sequence_length: int = 96,
        dl_epochs: int = 50,
        dl_batch_size: int = 64,
    ):
        """
        Initialize pipeline

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for outputs
            sequence_length: Sequence length for deep learning models (default: 96 = 24 hours)
            dl_epochs: Training epochs for deep learning models
            dl_batch_size: Batch size for deep learning models
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.sequence_length = sequence_length
        self.dl_epochs = dl_epochs
        self.dl_batch_size = dl_batch_size

        self.loader = DataLoader(data_dir)
        self.engineer = FeatureEngineer()
        self.evaluator = Evaluator()

        self.models = {}
        self.results = {}

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
        print("\n" + "=" * 80)
        print(f"LOADING AND PREPARING DATA FOR {site.upper()}")
        print("=" * 80)

        # Load training data
        train_data = self.loader.load_training_data(version=version)

        # Filter for specific site
        site_data = self.loader.get_site_data(site, train_data)

        # Create features
        print("\nCreating features...")
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

        print(f"\nData preparation complete!")
        print(f"Feature count: {len(feature_names)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        return (
            X_train,
            y_train_class,
            y_train_reg,
            X_val,
            y_val_class,
            y_val_reg,
            feature_names,
        )

    def train_all_models(
        self,
        X_train,
        y_train_class,
        y_train_reg,
        X_val,
        y_val_class,
        y_val_reg,
        tasks: list = ["classification", "regression"],
        models_type: str = "traditional",
    ):
        """
        Train models for specified tasks

        Args:
            X_train, y_train_class, y_train_reg: Training data
            X_val, y_val_class, y_val_reg: Validation data
            tasks: List of tasks to train for
            models_type: 'traditional', 'deep-learning', or 'all'
        """
        print("\n" + "=" * 80)
        print("TRAINING MODELS")
        print("=" * 80)

        # Traditional ML models
        traditional_models = {
            "XGBoost": XGBoostModel,
            "LightGBM": LightGBMModel,
            "CatBoost": CatBoostModel,
            "HistGradientBoosting": HistGradientBoostingModel,
        }

        # Deep learning models
        deep_learning_models = {}
        if DEEP_LEARNING_AVAILABLE:
            deep_learning_models = {
                "LSTM": LSTMModel,
                "GRU": GRUModel,
                "CNN": CNNModel,
                "TCN": TCNModel,
                "Transformer": TransformerModel,
            }

        # Select which models to train
        model_classes = {}
        if models_type in ["traditional", "all"]:
            model_classes.update(traditional_models)
        if models_type in ["deep-learning", "all"]:
            if not DEEP_LEARNING_AVAILABLE:
                print(
                    "\nWarning: Deep learning models requested but PyTorch not available!"
                )
                print("Install PyTorch with: pip install torch torchvision")
            else:
                model_classes.update(deep_learning_models)

        for task in tasks:
            print(f"\n{'='*80}")
            print(f"TASK: {task.upper()}")
            print(f"{'='*80}")

            y_train = y_train_class if task == "classification" else y_train_reg
            y_val = y_val_class if task == "classification" else y_val_reg

            for model_name, ModelClass in model_classes.items():
                print(f"\n{'='*80}")
                print(f"MODEL: {model_name} ({task})")
                print(f"{'='*80}")

                try:
                    # Create model with appropriate parameters
                    if model_name in deep_learning_models:
                        # Deep learning model
                        model = ModelClass(
                            task=task,
                            sequence_length=self.sequence_length,
                            epochs=self.dl_epochs,
                            batch_size=self.dl_batch_size,
                            early_stopping_patience=10,
                        )
                        # Convert to numpy arrays for deep learning
                        X_train_np = (
                            X_train.values if hasattr(X_train, "values") else X_train
                        )
                        X_val_np = X_val.values if hasattr(X_val, "values") else X_val
                        y_train_np = (
                            y_train.values if hasattr(y_train, "values") else y_train
                        )
                        y_val_np = y_val.values if hasattr(y_val, "values") else y_val

                        model.train(X_train_np, y_train_np, X_val_np, y_val_np)
                    else:
                        # Traditional ML model
                        model = ModelClass(task=task)
                        model.train(X_train, y_train, X_val, y_val)

                    # Store model
                    key = f"{model_name}_{task}"
                    self.models[key] = model

                    # Make predictions
                    if model_name in deep_learning_models:
                        train_pred = model.predict(X_train_np)
                        val_pred = model.predict(X_val_np)
                    else:
                        train_pred = model.predict(X_train)
                        val_pred = model.predict(X_val)

                    # Evaluate
                    if task == "classification":
                        train_results = self.evaluator.evaluate_classification(
                            y_train, train_pred, detailed=False
                        )
                        val_results = self.evaluator.evaluate_classification(
                            y_val, val_pred, detailed=True
                        )

                        print(f"\n{model_name} Results:")
                        print(f"Training F1: {train_results['f1_score_macro']:.4f}")
                        print(f"Validation F1: {val_results['f1_score_macro']:.4f}")
                        print(
                            f"Validation GM: {val_results['geometric_mean_score']:.4f}"
                        )
                    else:
                        train_results = self.evaluator.evaluate_regression(
                            y_train, train_pred, self.building_power_stats
                        )
                        val_results = self.evaluator.evaluate_regression(
                            y_val, val_pred, self.building_power_stats
                        )

                        print(f"\n{model_name} Results:")
                        print(f"Training MAE: {train_results['mae']:.4f}")
                        print(f"Validation MAE: {val_results['mae']:.4f}")
                        print(f"Validation RMSE: {val_results['rmse']:.4f}")
                        print(f"Validation CV-RMSE: {val_results['cv_rmse']:.4f}")

                    # Store results
                    self.results[key] = {
                        "train": train_results,
                        "validation": val_results,
                    }

                except Exception as e:
                    print(f"\nError training {model_name}: {str(e)}")
                    import traceback

                    traceback.print_exc()

    def evaluate_and_compare(self):
        """Evaluate and compare all models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        # Classification comparison
        print("\n--- CLASSIFICATION MODELS ---")
        class_results = {
            name.replace("_classification", ""): results["validation"]
            for name, results in self.results.items()
            if "classification" in name
        }

        if class_results:
            comparison_df = self.evaluator.compare_models(
                class_results, task="classification"
            )
            print(comparison_df.to_string(index=False))

        # Regression comparison
        print("\n--- REGRESSION MODELS ---")
        reg_results = {
            name.replace("_regression", ""): results["validation"]
            for name, results in self.results.items()
            if "regression" in name
        }

        if reg_results:
            comparison_df = self.evaluator.compare_models(
                reg_results, task="regression"
            )
            print(comparison_df.to_string(index=False))

    def save_results(self, site: str = "siteA", subfolder: str = None):
        """Save all results"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Save results JSON
        results_path = self.output_dir / f"results_{site}.json"
        save_results_json(self.results, str(results_path))

        # Create summary report
        report_path = self.output_dir / f"comparison_report_{site}.txt"
        all_results = {
            name: results["validation"] for name, results in self.results.items()
        }
        create_summary_report(all_results, str(report_path), task="both")

        # Save models
        if subfolder:
            models_dir = Path("../models") / subfolder
        else:
            models_dir = Path("../models")
        models_dir.mkdir(exist_ok=True, parents=True)

        for name, model in self.models.items():
            # Use .pth extension for PyTorch models, .pkl for others
            if any(
                dl_model in name
                for dl_model in ["LSTM", "GRU", "CNN", "TCN", "Transformer"]
            ):
                model_path = models_dir / f"{name}_{site}.pth"
            else:
                model_path = models_dir / f"{name}_{site}.pkl"
            model.save_model(str(model_path))

        print(f"\nResults saved to: {results_path}")
        print(f"Models saved to: {models_dir}")

    def prepare_merged_data(self, sites: list, version: str = "v0.2"):
        """Prepare merged data from multiple sites"""
        print("\n" + "=" * 80)
        print(f"PREPARING MERGED DATA FROM: {', '.join(sites)}")
        print("=" * 80)

        train_data = self.loader.load_training_data(version=version)
        merged_data = train_data[train_data["Site"].isin(sites)].copy()

        print(f"\nTotal samples: {len(merged_data)}")
        for site in sites:
            print(f"  {site}: {len(merged_data[merged_data['Site'] == site])} samples")

        # Create features
        print("\nCreating features...")
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

        print(
            f"\nFeatures: {len(feature_names)}, Train: {len(X_train)}, Val: {len(X_val)}"
        )
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
        models_type: str = "traditional",
    ):
        """Train separate models for each site"""
        print("\n" + "=" * 80)
        print(f"SITE-SPECIFIC TRAINING: {', '.join(sites)}")
        print("=" * 80)

        all_site_results = {}
        for site in sites:
            print(f"\n{'#'*80}\n# SITE: {site.upper()}\n{'#'*80}")
            X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, _ = (
                self.load_and_prepare_data(site=site, version=version)
            )

            self.models, self.results = {}, {}
            self.train_all_models(
                X_train,
                y_train_class,
                y_train_reg,
                X_val,
                y_val_class,
                y_val_reg,
                models_type=models_type,
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
        models_type: str = "traditional",
    ):
        """Train single merged model on all sites"""
        print("\n" + "=" * 80)
        print(f"MERGED MODEL TRAINING: {', '.join(sites)}")
        print("=" * 80)

        X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, _ = (
            self.prepare_merged_data(sites=sites, version=version)
        )

        self.models, self.results = {}, {}
        self.train_all_models(
            X_train,
            y_train_class,
            y_train_reg,
            X_val,
            y_val_class,
            y_val_reg,
            models_type=models_type,
        )
        self.evaluate_and_compare()
        self.save_results(site="merged", subfolder="merged")

        return self.models, self.results

    def _print_cross_site_comparison(self, all_site_results: dict):
        """Print cross-site comparison"""
        print("\n" + "=" * 80)
        print("CROSS-SITE PERFORMANCE COMPARISON")
        print("=" * 80)

        # Classification
        print("\n--- CLASSIFICATION ---")
        print(f"{'Model':<20} {'Site':<10} {'F1':>10} {'GM':>10}")
        print("-" * 60)
        for site, data in all_site_results.items():
            for name, res in data["results"].items():
                if "classification" in name:
                    vr = res["validation"]
                    print(
                        f"{name.replace('_classification', ''):<20} {site:<10} {vr['f1_score_macro']:>10.4f} {vr['geometric_mean_score']:>10.4f}"
                    )

        # Regression
        print("\n--- REGRESSION ---")
        print(f"{'Model':<20} {'Site':<10} {'MAE':>10} {'RMSE':>10} {'CV-RMSE':>10}")
        print("-" * 70)
        for site, data in all_site_results.items():
            for name, res in data["results"].items():
                if "regression" in name:
                    vr = res["validation"]
                    print(
                        f"{name.replace('_regression', ''):<20} {site:<10} {vr['mae']:>10.4f} {vr['rmse']:>10.4f} {vr['cv_rmse']:>10.4f}"
                    )

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
                    else:
                        f.write(
                            f"    MAE: {vr['mae']:.4f}, RMSE: {vr['rmse']:.4f}, CV-RMSE: {vr['cv_rmse']:.4f}\n"
                        )
        print(f"\nSaved to: {comp_path}")

    def run_full_pipeline(
        self,
        site: str = "siteA",
        version: str = "v0.2",
        models_type: str = "traditional",
        training_mode: str = "single",
    ):
        """
        Run the complete pipeline

        Args:
            site: Site to train on (for single mode)
            version: Data version
            models_type: 'traditional', 'deep-learning', or 'all'
            training_mode: 'single', 'site-specific', 'merged', or 'all'
        """
        print("\n" + "=" * 80)
        print("FLEXTRACK CHALLENGE - COMPLETE PIPELINE")
        print(f"Training Mode: {training_mode}, Models: {models_type}")
        print("=" * 80)

        sites = ["siteA", "siteB", "siteC"]

        if training_mode == "single":
            X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, _ = (
                self.load_and_prepare_data(site=site, version=version)
            )
            self.train_all_models(
                X_train,
                y_train_class,
                y_train_reg,
                X_val,
                y_val_class,
                y_val_reg,
                models_type=models_type,
            )
            self.evaluate_and_compare()
            self.save_results(site=site)

        elif training_mode == "site-specific":
            self.run_site_specific_training(
                sites=sites, version=version, models_type=models_type
            )

        elif training_mode == "merged":
            self.run_merged_training(
                sites=sites, version=version, models_type=models_type
            )

        elif training_mode == "all":
            print("\n" + "=" * 80 + "\nCOMPREHENSIVE TRAINING\n" + "=" * 80)
            self.run_site_specific_training(
                sites=sites, version=version, models_type=models_type
            )
            self.run_merged_training(
                sites=sites, version=version, models_type=models_type
            )
            print(f"\n✓ Trained {len(sites)} site-specific + 1 merged model")
            print(f"✓ Results: results/, Models: models/")

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)


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
        "--version", type=str, default="v0.2", help="Data version (default: v0.2)"
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
        "--models-type",
        type=str,
        default="traditional",
        choices=["traditional", "deep-learning", "all"],
        help="Type of models to train (default: traditional)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=96,
        help="Sequence length for deep learning models (default: 96)",
    )
    parser.add_argument(
        "--dl-epochs",
        type=int,
        default=50,
        help="Training epochs for deep learning (default: 50)",
    )
    parser.add_argument(
        "--dl-batch-size",
        type=int,
        default=64,
        help="Batch size for deep learning (default: 64)",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="single",
        choices=["single", "site-specific", "merged", "all"],
        help="Training mode: single site, site-specific, merged, or all (default: single)",
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = FlexTrackPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        dl_epochs=args.dl_epochs,
        dl_batch_size=args.dl_batch_size,
    )

    pipeline.run_full_pipeline(
        site=args.site,
        version=args.version,
        models_type=args.models_type,
        training_mode=args.training_mode,
    )


if __name__ == "__main__":
    main()
