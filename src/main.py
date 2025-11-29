import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Union

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_VERSION = "v0.2"

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
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import pandas as pd
import numpy as np


class FlexTrackPipeline:

    def __init__(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        output_dir: Union[str, Path] = DEFAULT_RESULTS_DIR,
        sampler: str = "none",
        tasks: Optional[List[str]] = None,
        use_advanced_weights: bool = True,
        use_ensemble: bool = True,
        use_cascade: bool = True,
        feature_selection: bool = True,
        n_features: int = 80,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sampler = sampler
        self.use_advanced_weights = use_advanced_weights
        self.use_ensemble = use_ensemble
        self.use_cascade = use_cascade
        self.feature_selection = feature_selection
        self.n_features = n_features

        if tasks is None:
            tasks = ["classification", "regression"]
        self.tasks = list(dict.fromkeys(tasks))

        allowed_tasks = {"classification", "regression"}
        unknown_tasks = set(self.tasks) - allowed_tasks
        if unknown_tasks:
            raise ValueError(f"Unknown task(s): {', '.join(sorted(unknown_tasks))}")
        if not self.tasks:
            raise ValueError("At least one task must be specified")

        self.loader = DataLoader(data_dir)
        self.engineer = FeatureEngineer()
        self.evaluator = Evaluator()
        self.models = {}
        self.results = {}
        self.selected_features = None

    def load_and_prepare_data(self, site: str = "siteA", version: str = "v0.2"):
        train_data = self.loader.load_training_data(version=version)
        site_data = self.loader.get_site_data(site, train_data)
        site_data_features = self.engineer.create_all_features(
            site_data, include_lags=True, include_rolling=True
        )
        train_df, val_df = self.loader.split_train_validation(
            site_data_features, validation_size=0.2, time_based=True
        )
        feature_names = self.engineer.get_feature_names()
        X_train = train_df[feature_names]
        y_train_class = train_df["Demand_Response_Flag"]
        y_train_reg = train_df["Demand_Response_Capacity_kW"]
        X_val = val_df[feature_names]
        y_val_class = val_df["Demand_Response_Flag"]
        y_val_reg = val_df["Demand_Response_Capacity_kW"]
        building_power = site_data["Building_Power_kW"]
        self.building_power_stats = self.evaluator.calculate_building_power_stats(building_power)
        return X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names

    def select_top_features(self, X_train, y_train, n_features=None):
        if n_features is None:
            n_features = self.n_features
        from sklearn.ensemble import RandomForestClassifier
        y_adjusted = y_train + 1
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        clf.fit(X_train, y_adjusted)
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        selected = importances.head(n_features)['feature'].tolist()
        self.selected_features = selected
        return selected

    def compute_dynamic_class_weights(self, y_train, extreme_penalty: bool = False):
        y_adjusted = y_train + 1
        unique_classes, counts = np.unique(y_adjusted, return_counts=True)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights_array = (1.0 - beta) / effective_num
        if extreme_penalty:
            max_count = counts.max()
            minority_mask = counts < (max_count * 0.1)
            weights_array[minority_mask] *= 5.0
        weights = {cls - 1: w for cls, w in zip(unique_classes, weights_array)}
        return weights

    def apply_resampling(self, X_train, y_train, method='smote'):
        y_adjusted = y_train + 1
        _, counts = np.unique(y_adjusted, return_counts=True)
        min_samples = min(counts)
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        majority_count = counts.max()
        sampling_strategy = {0: majority_count, 2: majority_count}

        try:
            if method == 'smote':
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            elif method == 'adasyn':
                sampler = ADASYN(random_state=42, n_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            elif method == 'smoteenn':
                sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy))
            elif method == 'borderline':
                sampler = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, kind='borderline-1')
            elif method == 'svmsmote':
                sampler = SVMSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            elif method == 'kmeans':
                sampler = KMeansSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, cluster_balance_threshold=0.01)
            elif method == 'smotetomek':
                sampler = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy)
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            X_resampled, y_resampled = sampler.fit_resample(X_train, y_adjusted)
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled - 1, name=y_train.name)
            return X_resampled, y_resampled
        except Exception:
            return X_train, y_train

    def train_all_models(self, X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, tasks: Optional[List[str]] = None):
        if "classification" in (tasks or self.tasks) and self.feature_selection and len(X_train.columns) > self.n_features:
            selected_features = self.select_top_features(X_train, y_train_class, self.n_features)
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]

        model_classes = {
            "XGBoost": XGBoostModel,
            "LightGBM": LightGBMModel,
            "CatBoost": CatBoostModel,
            "HistGradientBoosting": HistGradientBoostingModel,
        }

        tasks = tasks or self.tasks

        for task in tasks:
            y_train = y_train_class if task == "classification" else y_train_reg
            y_val = y_val_class if task == "classification" else y_val_reg

            if task == "classification" and self.sampler != "none":
                X_train_balanced, y_train_balanced = self.apply_resampling(X_train, y_train, method=self.sampler)
            else:
                X_train_balanced, y_train_balanced = X_train, y_train

            if task == "classification" and self.use_advanced_weights:
                class_weights = self.compute_dynamic_class_weights(y_train_balanced)
                weight_dict_adjusted = {k + 1: v for k, v in class_weights.items()}
            else:
                weight_dict_adjusted = None

            for model_name, ModelClass in model_classes.items():
                print(f"Training {model_name} ({task})...")
                try:
                    if task == "classification" and weight_dict_adjusted is not None:
                        if model_name == "LightGBM":
                            model = ModelClass(task=task)
                            if hasattr(model, 'params'):
                                model.params['class_weight'] = weight_dict_adjusted
                        else:
                            model = ModelClass(task=task)
                    else:
                        model = ModelClass(task=task)

                    model.train(X_train_balanced, y_train_balanced, X_val, y_val)
                    key = f"{model_name}_{task}"
                    self.models[key] = model

                    if task == "classification":
                        train_pred = model.predict(X_train)
                        val_pred = model.predict(X_val)
                        train_results = self.evaluator.evaluate_classification(y_train, train_pred, detailed=False)
                        val_results = self.evaluator.evaluate_classification(y_val, val_pred, detailed=True)
                        print(f"  {model_name}: Train F1={train_results['f1_score_macro']:.4f}, Val F1={val_results['f1_score_macro']:.4f}, Val GM={val_results['geometric_mean_score']:.4f}")
                    else:
                        train_pred = model.predict(X_train)
                        val_pred = model.predict(X_val)
                        train_results = self.evaluator.evaluate_regression(y_train, train_pred, self.building_power_stats)
                        val_results = self.evaluator.evaluate_regression(y_val, val_pred, self.building_power_stats)
                        cv_rmse_str = f", CV-RMSE={val_results['cv_rmse']:.4f}" if 'cv_rmse' in val_results else ""
                        print(f"  {model_name}: Train MAE={train_results['mae']:.4f}, Val MAE={val_results['mae']:.4f}, Val RMSE={val_results['rmse']:.4f}{cv_rmse_str}")

                    self.results[key] = {"train": train_results, "validation": val_results}
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")

            if task == "classification" and self.use_ensemble:
                print("\nCreating Ensemble Model (Weighted Voting)...")
                try:
                    model_probas = []
                    model_f1s = []
                    for model_name in ["XGBoost", "LightGBM", "CatBoost"]:
                        key = f"{model_name}_{task}"
                        if key in self.models and key in self.results:
                            model = self.models[key]
                            val_proba = model.predict_proba(X_val)
                            model_probas.append(val_proba)
                            f1 = self.results[key]["validation"]["f1_score_macro"]
                            model_f1s.append(f1)

                    if len(model_probas) >= 2:
                        weights = np.array(model_f1s) / sum(model_f1s)
                        ensemble_proba = np.average(model_probas, axis=0, weights=weights)
                        val_pred = np.argmax(ensemble_proba, axis=1) - 1
                        val_results = self.evaluator.evaluate_classification(y_val, val_pred, detailed=True)

                        class EnsembleModel:
                            def __init__(self, models, weights):
                                self.models = models
                                self.weights = weights
                            def predict_proba(self, X):
                                probas = [m.predict_proba(X) for m in self.models]
                                return np.average(probas, axis=0, weights=self.weights)
                            def predict(self, X):
                                proba = self.predict_proba(X)
                                return np.argmax(proba, axis=1) - 1
                            def save_model(self, filepath):
                                import joblib
                                joblib.dump({'weights': self.weights, 'model_names': ['XGBoost', 'LightGBM', 'CatBoost']}, filepath)

                        ensemble_models = [self.models[f"{name}_{task}"] for name in ["XGBoost", "LightGBM", "CatBoost"] if f"{name}_{task}" in self.models]
                        ensemble = EnsembleModel(ensemble_models, weights)
                        key = "Ensemble_classification"
                        self.models[key] = ensemble
                        self.results[key] = {"train": {}, "validation": val_results}
                        print(f"  Ensemble: Val F1={val_results['f1_score_macro']:.4f}, Val GM={val_results['geometric_mean_score']:.4f}")
                        print(f"  Weights: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f}")
                except Exception as e:
                    print(f"Error creating ensemble: {str(e)}")

            if task == "classification" and self.use_cascade:
                print("\nTraining CASCADE Classifier (Hierarchical)...")
                try:
                    cascade = CascadeClassifier(stage1_model="xgboost", stage2_model="lightgbm")
                    cascade.train(X_train_balanced, y_train_balanced, X_val, y_val)
                    val_pred = cascade.predict(X_val)
                    val_results = self.evaluator.evaluate_classification(y_val, val_pred, detailed=True)
                    key = "Cascade_classification"
                    self.models[key] = cascade
                    self.results[key] = {"train": {}, "validation": val_results}
                    print(f"  Cascade: Val F1={val_results['f1_score_macro']:.4f}, Val GM={val_results['geometric_mean_score']:.4f}")
                except Exception as e:
                    print(f"Error training cascade: {str(e)}")

    def evaluate_and_compare(self):
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
                comparison_df = self.evaluator.compare_models(class_results, task="classification")
                print("\nCLASSIFICATION MODELS:")
                print(comparison_df.to_string(index=False))

        if "regression" in self.tasks:
            reg_results = {
                name.replace("_regression", ""): results["validation"]
                for name, results in self.results.items()
                if "regression" in name
            }
            if reg_results:
                comparison_df = self.evaluator.compare_models(reg_results, task="regression")
                print("\nREGRESSION MODELS:")
                print(comparison_df.to_string(index=False))

    def save_results(self, site: str = "siteA", subfolder: str = None):
        print("\nSaving results...")
        results_path = self.output_dir / f"results_{site}.json"
        save_results_json(self.results, str(results_path))
        report_path = self.output_dir / f"comparison_report_{site}.txt"
        all_results = {name: results["validation"] for name, results in self.results.items()}
        create_summary_report(all_results, str(report_path), task=self._get_summary_task_mode())

        if subfolder:
            models_dir = Path("../models") / subfolder
        else:
            models_dir = Path("../models")
        models_dir.mkdir(exist_ok=True, parents=True)

        for name, model in self.models.items():
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
        train_data = self.loader.load_training_data(version=version)
        merged_data = train_data[train_data["Site"].isin(sites)].copy()
        merged_data_features = self.engineer.create_all_features(merged_data, include_lags=True, include_rolling=True)
        train_df, val_df = self.loader.split_train_validation(merged_data_features, validation_size=0.2, time_based=True)
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
        self.building_power_stats = self.evaluator.calculate_building_power_stats(merged_data["Building_Power_kW"])
        return X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names

    def run_site_specific_training(self, sites: list = ["siteA", "siteB", "siteC"], version: str = "v0.2"):
        all_site_results = {}
        for site in sites:
            print(f"\n{'='*40} SITE: {site.upper()} {'='*40}")
            X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names = (
                self.load_and_prepare_data(site=site, version=version)
            )
            self.feature_names = feature_names
            self.models, self.results = {}, {}
            self.train_all_models(X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, tasks=self.tasks)
            all_site_results[site] = {"models": self.models.copy(), "results": self.results.copy()}
            self.save_results(site=site, subfolder=f"site_specific/{site}")
        self._print_cross_site_comparison(all_site_results)
        return all_site_results

    def run_merged_training(self, sites: list = ["siteA", "siteB", "siteC"], version: str = "v0.2"):
        X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names = (
            self.prepare_merged_data(sites=sites, version=version)
        )
        self.feature_names = feature_names
        self.models, self.results = {}, {}
        self.train_all_models(X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, tasks=self.tasks)
        self.evaluate_and_compare()
        self.save_results(site="merged", subfolder="merged")
        return self.models, self.results

    def _print_cross_site_comparison(self, all_site_results: dict):
        comp_path = self.output_dir / "cross_site_comparison.txt"
        with open(comp_path, "w") as f:
            f.write("CROSS-SITE PERFORMANCE COMPARISON\n" + "=" * 80 + "\n")
            for site, data in all_site_results.items():
                f.write(f"\n{site.upper()}:\n")
                for name, res in data["results"].items():
                    vr = res["validation"]
                    f.write(f"  {name}:\n")
                    if "classification" in name:
                        f.write(f"    F1: {vr['f1_score_macro']:.4f}, GM: {vr['geometric_mean_score']:.4f}\n")
                    elif "regression" in name:
                        cv_rmse_val = vr.get('cv_rmse', 0.0)
                        f.write(f"    MAE: {vr['mae']:.4f}, RMSE: {vr['rmse']:.4f}, CV-RMSE: {cv_rmse_val:.4f}\n")

    def run_full_pipeline(self, site: str = "siteA", version: str = "v0.2", training_mode: str = "single"):
        sites = ["siteA", "siteB", "siteC"]

        if training_mode == "single":
            X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg, feature_names = (
                self.load_and_prepare_data(site=site, version=version)
            )
            self.feature_names = feature_names
            self.train_all_models(X_train, y_train_class, y_train_reg, X_val, y_val_class, y_val_reg)
            self.evaluate_and_compare()
            self.save_results(site=site)
        elif training_mode == "site-specific":
            self.run_site_specific_training(sites=sites, version=version)
        elif training_mode == "merged":
            self.run_merged_training(sites=sites, version=version)
        elif training_mode == "all":
            self.run_site_specific_training(sites=sites, version=version)
            self.run_merged_training(sites=sites, version=version)

        print("\nPipeline completed successfully!")


def main():
    """
    FlexTrack Challenge - Energy Flexibility Prediction

    Train machine learning models for demand response classification and regression.

    Usage:
        # Single site with all optimizations
        python main.py --site siteB --tasks classification

        # All sites with specific optimizations
        python main.py --training-mode site-specific --use-cascade --use-ensemble

        # Disable specific features
        python main.py --no-cascade --no-ensemble --no-feature-selection
    """
    parser = argparse.ArgumentParser(
        description="FlexTrack Challenge - Energy Flexibility Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--site", type=str, default="siteA", help="Site to train on (default: siteA)")
    parser.add_argument("--version", type=str, default="v0.2", help="Data version (default: v0.2)")
    parser.add_argument("--data-dir", type=str, default="../data", help="Data directory (default: ../data)")
    parser.add_argument("--output-dir", type=str, default="../results", help="Output directory (default: ../results)")
    parser.add_argument(
        "--training-mode",
        type=str,
        default="single",
        choices=["single", "site-specific", "merged", "all"],
        help="Training mode (default: single)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="none",
        choices=["none", "smote", "adasyn", "smoteenn", "borderline", "svmsmote", "kmeans", "smotetomek"],
        help="Sampling method for class imbalance (default: none)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["classification", "regression"],
        default=["classification", "regression"],
        help="Tasks to run (default: both)",
    )
    parser.add_argument("--use-advanced-weights", action="store_true", default=True, help="Use dynamic class weights (default: True)")
    parser.add_argument("--no-advanced-weights", dest="use_advanced_weights", action="store_false", help="Disable advanced class weighting")
    parser.add_argument("--use-ensemble", action="store_true", default=True, help="Enable ensemble voting (default: True)")
    parser.add_argument("--no-ensemble", dest="use_ensemble", action="store_false", help="Disable ensemble voting")
    parser.add_argument("--use-cascade", action="store_true", default=True, help="Enable cascade classifier (default: True)")
    parser.add_argument("--no-cascade", dest="use_cascade", action="store_false", help="Disable cascade classifier")
    parser.add_argument("--feature-selection", action="store_true", default=True, help="Enable feature selection (default: True)")
    parser.add_argument("--no-feature-selection", dest="feature_selection", action="store_false", help="Disable feature selection")
    parser.add_argument("--n-features", type=int, default=80, help="Number of features to select (default: 80)")

    args = parser.parse_args()

    pipeline = FlexTrackPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampler=args.sampler,
        tasks=args.tasks,
        use_advanced_weights=args.use_advanced_weights,
        use_ensemble=args.use_ensemble,
        use_cascade=args.use_cascade,
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
