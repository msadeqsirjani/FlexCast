import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, root_mean_squared_error, confusion_matrix
)
from typing import Dict
import warnings

warnings.filterwarnings("ignore")


class Evaluator:

    def __init__(self):
        pass

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, detailed: bool = True) -> Dict[str, float]:
        labels = [-1, 0, 1]
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels)
        f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels)
        precision_macro = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        precision_micro = precision_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        gm_score = geometric_mean_score(y_true, y_pred, average="macro")
        results = {
            "accuracy": accuracy, "f1_score_macro": f1_macro, "f1_score_micro": f1_micro,
            "f1_score_weighted": f1_weighted, "precision_macro": precision_macro,
            "precision_micro": precision_micro, "precision_weighted": precision_weighted,
            "recall_macro": recall_macro, "recall_micro": recall_micro,
            "recall_weighted": recall_weighted, "geometric_mean_score": gm_score,
        }
        if detailed:
            f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
            for i, class_label in enumerate(labels):
                results[f"f1_score_class_{class_label}"] = f1_per_class[i]
            precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
            for i, class_label in enumerate(labels):
                results[f"precision_class_{class_label}"] = precision_per_class[i]
            recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
            for i, class_label in enumerate(labels):
                results[f"recall_class_{class_label}"] = recall_per_class[i]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            results["confusion_matrix"] = cm
        return results

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, building_power_stats: Dict[str, float] = None) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        results = {"mae": mae, "rmse": rmse}
        if building_power_stats is not None:
            power_range = building_power_stats.get("range")
            power_mean = building_power_stats.get("mean")
            if power_range and power_range > 0:
                results["nmae_range"] = (mae / power_range) * 100
                results["nrmse_range"] = (rmse / power_range) * 100
            if power_mean and power_mean > 0:
                results["nmae_mean"] = (mae / power_mean) * 100
                results["nrmse_mean"] = (rmse / power_mean) * 100
        y_nonzero = y_true[y_true != 0]
        if len(y_nonzero) > 0:
            y_mean = np.abs(np.mean(y_nonzero))
            if y_mean > 0:
                results["cv_rmse"] = (rmse / y_mean) * 100
        return results

    def evaluate_combined(self, y_true_class: np.ndarray, y_pred_class: np.ndarray, y_true_reg: np.ndarray, y_pred_reg: np.ndarray, building_power_stats: Dict[str, float] = None) -> Dict[str, Dict]:
        class_results = self.evaluate_classification(y_true_class, y_pred_class)
        reg_results = self.evaluate_regression(y_true_reg, y_pred_reg, building_power_stats)
        return {"classification": class_results, "regression": reg_results}

    def calculate_building_power_stats(self, building_power: pd.Series) -> Dict[str, float]:
        non_zero = building_power[building_power != 0]
        if len(non_zero) == 0:
            return {"min": 0, "max": 0, "mean": 0, "range": 0}
        min_val = non_zero.min()
        max_val = non_zero.max()
        return {"min": min_val, "max": max_val, "mean": non_zero.mean(), "range": max_val - min_val}

    def compare_models(self, model_results: Dict[str, Dict], task: str = "classification") -> pd.DataFrame:
        comparison_data = []
        for model_name, results in model_results.items():
            if task == "classification":
                comparison_data.append({
                    "Model": model_name,
                    "F1 Score": results.get("f1_score_macro", np.nan),
                    "Geometric Mean": results.get("geometric_mean_score", np.nan),
                })
            else:
                comparison_data.append({
                    "Model": model_name,
                    "MAE": results.get("mae", np.nan),
                    "RMSE": results.get("rmse", np.nan),
                    "NMAE (range)": results.get("nmae_range", np.nan),
                    "CV-RMSE": results.get("cv_rmse", np.nan),
                })
        df = pd.DataFrame(comparison_data)
        if task == "classification":
            df = df.sort_values("F1 Score", ascending=False)
        else:
            df = df.sort_values("MAE", ascending=True)
        return df
