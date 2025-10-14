"""
Data Loader Module for FlexTrack Challenge 2025
Handles loading and preprocessing of building energy data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class DataLoader:
    """Load and preprocess FlexTrack Challenge 2025 data"""

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize DataLoader

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.training_data = None
        self.test_data = None

    def load_training_data(self, version: str = "v0.2") -> pd.DataFrame:
        """
        Load training data

        Args:
            version: Data version to load (v0.1 or v0.2)

        Returns:
            DataFrame with training data
        """
        pattern = f"*flextrack-2025-training-data-{version}.csv"
        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No training data found matching {pattern}")

        print(f"Loading training data: {files[0].name}")
        self.training_data = pd.read_csv(files[0])

        # Parse timestamp
        self.training_data["Timestamp_Local"] = pd.to_datetime(
            self.training_data["Timestamp_Local"]
        )

        print(f"Loaded {len(self.training_data)} training samples")
        print(f"Sites: {self.training_data['Site'].unique()}")
        print(
            f"Date range: {self.training_data['Timestamp_Local'].min()} to {self.training_data['Timestamp_Local'].max()}"
        )

        return self.training_data

    def load_test_data(self, version: str = "v0.2") -> pd.DataFrame:
        """
        Load test data

        Args:
            version: Data version to load (v0.1 or v0.2)

        Returns:
            DataFrame with test data
        """
        pattern = f"*flextrack-2025-public-test-data-{version}.csv"
        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No test data found matching {pattern}")

        print(f"Loading test data: {files[0].name}")
        self.test_data = pd.read_csv(files[0])

        # Parse timestamp
        self.test_data["Timestamp_Local"] = pd.to_datetime(
            self.test_data["Timestamp_Local"]
        )

        print(f"Loaded {len(self.test_data)} test samples")

        return self.test_data

    def get_site_data(
        self, site: str, data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Filter data for a specific site

        Args:
            site: Site identifier (e.g., 'siteA')
            data: DataFrame to filter (uses training_data if None)

        Returns:
            Filtered DataFrame
        """
        if data is None:
            data = self.training_data

        if data is None:
            raise ValueError("No data loaded. Call load_training_data() first.")

        site_data = data[data["Site"] == site].copy()
        print(f"Site {site}: {len(site_data)} samples")

        return site_data

    def split_train_validation(
        self, data: pd.DataFrame, validation_size: float = 0.2, time_based: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets

        Args:
            data: DataFrame to split
            validation_size: Fraction of data for validation
            time_based: If True, use time-based split (last validation_size fraction)

        Returns:
            Tuple of (train_df, val_df)
        """
        if time_based:
            # Sort by timestamp
            data = data.sort_values("Timestamp_Local")
            split_idx = int(len(data) * (1 - validation_size))
            train_df = data.iloc[:split_idx].copy()
            val_df = data.iloc[split_idx:].copy()
        else:
            # Random split
            from sklearn.model_selection import train_test_split

            train_df, val_df = train_test_split(
                data, test_size=validation_size, random_state=42
            )

        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")

        return train_df, val_df

    def get_data_statistics(self, data: pd.DataFrame) -> dict:
        """
        Get statistics about the dataset

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary with statistics
        """
        stats = {
            "num_samples": len(data),
            "num_sites": data["Site"].nunique(),
            "sites": data["Site"].unique().tolist(),
            "date_range": (
                data["Timestamp_Local"].min(),
                data["Timestamp_Local"].max(),
            ),
            "dr_flag_distribution": (
                data["Demand_Response_Flag"].value_counts().to_dict()
                if "Demand_Response_Flag" in data.columns
                else None
            ),
            "missing_values": data.isnull().sum().to_dict(),
            "power_stats": {
                "mean": data["Building_Power_kW"].mean(),
                "std": data["Building_Power_kW"].std(),
                "min": data["Building_Power_kW"].min(),
                "max": data["Building_Power_kW"].max(),
            },
        }

        if "Demand_Response_Capacity_kW" in data.columns:
            stats["capacity_stats"] = {
                "mean": data["Demand_Response_Capacity_kW"].mean(),
                "std": data["Demand_Response_Capacity_kW"].std(),
                "min": data["Demand_Response_Capacity_kW"].min(),
                "max": data["Demand_Response_Capacity_kW"].max(),
            }

        return stats

    def handle_missing_values(
        self, data: pd.DataFrame, method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            data: DataFrame with potential missing values
            method: Method to use ('ffill', 'bfill', 'interpolate', 'drop')

        Returns:
            DataFrame with missing values handled
        """
        data = data.copy()

        if method == "ffill":
            data = data.fillna(method="ffill")
        elif method == "bfill":
            data = data.fillna(method="bfill")
        elif method == "interpolate":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate(method="linear")
        elif method == "drop":
            data = data.dropna()

        # Fill any remaining NaN with 0
        data = data.fillna(0)

        return data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader("../data")

    # Load training data
    train_data = loader.load_training_data(version="v0.2")

    # Get statistics
    stats = loader.get_data_statistics(train_data)
    print("\nData Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Get site-specific data
    site_a_data = loader.get_site_data("siteA")

    # Split into train/validation
    train_df, val_df = loader.split_train_validation(site_a_data)
