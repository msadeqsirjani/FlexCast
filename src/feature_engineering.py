"""
Feature Engineering Module for FlexTrack Challenge 2025
Creates temporal, statistical, and lag features from time-series data
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
import logging

warnings.filterwarnings("ignore")

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                  datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FeatureEngineer:
    """Create features for demand response prediction"""

    def __init__(self):
        """Initialize FeatureEngineer"""
        self.feature_names = []

    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp

        Args:
            data: DataFrame with 'Timestamp_Local' column

        Returns:
            DataFrame with temporal features added
        """
        data = data.copy()

        # Extract temporal components
        data["hour"] = data["Timestamp_Local"].dt.hour
        data["day_of_week"] = data["Timestamp_Local"].dt.dayofweek
        data["day_of_month"] = data["Timestamp_Local"].dt.day
        data["month"] = data["Timestamp_Local"].dt.month
        data["quarter"] = data["Timestamp_Local"].dt.quarter
        data["day_of_year"] = data["Timestamp_Local"].dt.dayofyear
        data["week_of_year"] = data["Timestamp_Local"].dt.isocalendar().week.astype(int)

        # Cyclical encoding for periodic features
        # Hour (24-hour cycle)
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

        # Day of week (7-day cycle)
        data["day_of_week_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_of_week_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # Month (12-month cycle)
        data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
        data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

        # Day of year (365-day cycle)
        data["day_of_year_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365)
        data["day_of_year_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365)

        # Binary features
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
        data["is_business_hours"] = ((data["hour"] >= 9) & (data["hour"] <= 17)).astype(
            int
        )
        data["is_peak_hours"] = ((data["hour"] >= 17) & (data["hour"] <= 21)).astype(
            int
        )
        data["is_night"] = ((data["hour"] >= 22) | (data["hour"] <= 6)).astype(int)

        logger.info("Created 21 temporal features")

        return data

    def create_lag_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 4, 8, 12, 24, 48, 96],
    ) -> pd.DataFrame:
        """
        Create lag features for time-series data

        Args:
            data: DataFrame to create lags from
            columns: Columns to create lags for
            lags: List of lag periods (in 15-min intervals)

        Returns:
            DataFrame with lag features added
        """
        data = data.copy()
        lag_features = []

        for col in columns:
            if col not in data.columns:
                continue

            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                data[lag_col] = data.groupby("Site")[col].shift(lag)
                lag_features.append(lag_col)

        logger.info(f"Created {len(lag_features)} lag features")

        return data

    def create_rolling_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [4, 8, 12, 24, 48, 96],
    ) -> pd.DataFrame:
        """
        Create rolling window statistical features

        Args:
            data: DataFrame to create rolling features from
            columns: Columns to create rolling features for
            windows: Window sizes (in 15-min intervals)

        Returns:
            DataFrame with rolling features added
        """
        data = data.copy()
        rolling_features = []

        for col in columns:
            if col not in data.columns:
                continue

            for window in windows:
                # Rolling mean
                roll_mean_col = f"{col}_rolling_mean_{window}"
                data[roll_mean_col] = data.groupby("Site")[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                rolling_features.append(roll_mean_col)

                # Rolling std
                roll_std_col = f"{col}_rolling_std_{window}"
                data[roll_std_col] = data.groupby("Site")[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                rolling_features.append(roll_std_col)

                # Rolling min
                roll_min_col = f"{col}_rolling_min_{window}"
                data[roll_min_col] = data.groupby("Site")[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                rolling_features.append(roll_min_col)

                # Rolling max
                roll_max_col = f"{col}_rolling_max_{window}"
                data[roll_max_col] = data.groupby("Site")[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                rolling_features.append(roll_max_col)

        logger.info(f"Created {len(rolling_features)} rolling features")

        return data

    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables

        Args:
            data: DataFrame to create interactions from

        Returns:
            DataFrame with interaction features added
        """
        data = data.copy()

        # Temperature-related interactions
        if "Dry_Bulb_Temperature_C" in data.columns:
            # Temperature squared (non-linear effect)
            data["temp_squared"] = data["Dry_Bulb_Temperature_C"] ** 2

            # Temperature categories
            data["temp_category"] = pd.cut(
                data["Dry_Bulb_Temperature_C"],
                bins=[-np.inf, 10, 20, 30, np.inf],
                labels=[0, 1, 2, 3],
            ).astype(int)

        # Solar radiation interactions
        if "Global_Horizontal_Radiation_W/m2" in data.columns:
            # Solar radiation squared
            data["solar_squared"] = data["Global_Horizontal_Radiation_W/m2"] ** 2

            # Is daylight
            data["is_daylight"] = (
                data["Global_Horizontal_Radiation_W/m2"] > 50
            ).astype(int)

        # Temperature and solar interaction
        if (
            "Dry_Bulb_Temperature_C" in data.columns
            and "Global_Horizontal_Radiation_W/m2" in data.columns
        ):
            data["temp_solar_interaction"] = (
                data["Dry_Bulb_Temperature_C"]
                * data["Global_Horizontal_Radiation_W/m2"]
            )

        # Building power features
        if "Building_Power_kW" in data.columns:
            # Power squared
            data["power_squared"] = data["Building_Power_kW"] ** 2

            # Power rate of change (key for DR prediction!)
            data["power_rate_change"] = data.groupby("Site")["Building_Power_kW"].diff()
            
            # Power acceleration (2nd derivative)
            data["power_acceleration"] = data.groupby("Site")["power_rate_change"].diff()
            
            # Power volatility (absolute rate of change)
            data["power_volatility"] = data["power_rate_change"].abs()
            
            # Power deviation from rolling mean (anomaly detection)
            data["power_deviation_24h"] = data["Building_Power_kW"] - data.groupby("Site")["Building_Power_kW"].transform(
                lambda x: x.rolling(window=96, min_periods=1).mean()  # 24 hours
            )
            
            # Power percentile within rolling window
            data["power_percentile_24h"] = data.groupby("Site")["Building_Power_kW"].transform(
                lambda x: x.rolling(window=96, min_periods=1).apply(
                    lambda y: (y.iloc[-1] <= y).sum() / len(y) if len(y) > 0 else 0.5
                )
            )

        # DR Capacity lag features (if available in training)
        if "Demand_Response_Capacity_kW" in data.columns:
            # Historical DR activity
            data["dr_capacity_lag_96"] = data.groupby("Site")["Demand_Response_Capacity_kW"].shift(96)  # Same time yesterday
            data["dr_capacity_lag_672"] = data.groupby("Site")["Demand_Response_Capacity_kW"].shift(672)  # Same time last week

        logger.info("Created interaction features")

        return data

    def create_energy_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create energy-specific pattern features for demand response prediction
        
        Args:
            data: DataFrame with power and temporal features
            
        Returns:
            DataFrame with energy pattern features added
        """
        data = data.copy()
        
        if "Building_Power_kW" in data.columns:
            # Peak vs off-peak consumption patterns
            # Define peak hours (typically 17:00-21:00)
            if "hour" in data.columns:
                peak_mask = data["hour"].between(17, 21)
                peak_power = data["Building_Power_kW"].where(peak_mask, 0.0)
                data["avg_peak_power_1week"] = (
                    peak_power.groupby(data["Site"]).transform(
                        lambda s: s.rolling(window=672, min_periods=1).mean()
                    )
                )

                offpeak_mask = ~peak_mask
                offpeak_power = data["Building_Power_kW"].where(offpeak_mask, 0.0)
                data["avg_offpeak_power_1week"] = (
                    offpeak_power.groupby(data["Site"]).transform(
                        lambda s: s.rolling(window=672, min_periods=1).mean()
                    )
                )

                # Peak-to-offpeak ratio (clip to prevent extreme values)
                data["peak_offpeak_ratio"] = (data["avg_peak_power_1week"] / 
                                             np.maximum(data["avg_offpeak_power_1week"], 1e-3)).clip(0, 1000)
            
            # Load factor (average / max over rolling window)
            data["load_factor_24h"] = data.groupby("Site")["Building_Power_kW"].transform(
                lambda x: (x.rolling(window=96, min_periods=1).mean() / 
                          np.maximum(x.rolling(window=96, min_periods=1).max(), 1e-3)).clip(0, 1)
            )
            
            # Power ramp rate (max change in rolling window)
            data["max_ramp_rate_4h"] = data.groupby("Site")["power_rate_change"].transform(
                lambda x: x.rolling(window=16, min_periods=1).max().abs()
            )
            
            # Demand Response opportunity score (high power + high volatility = good DR candidate)
            if "power_volatility" in data.columns:
                # Normalize both features to 0-1 range per site
                data["power_normalized"] = data.groupby("Site")["Building_Power_kW"].transform(
                    lambda x: ((x - x.min()) / np.maximum(x.max() - x.min(), 1e-3)).clip(0, 1)
                )
                data["volatility_normalized"] = data.groupby("Site")["power_volatility"].transform(
                    lambda x: ((x - x.min()) / np.maximum(x.max() - x.min(), 1e-3)).clip(0, 1)
                )
                data["dr_opportunity_score"] = data["power_normalized"] * data["volatility_normalized"]
        
        # Weather-power correlation patterns
        if "Dry_Bulb_Temperature_C" in data.columns and "Building_Power_kW" in data.columns:
            # Temperature-power sensitivity (rolling correlation)
            window = 96  # 24 hours
            # Compute rolling correlation per site
            correlations = []
            for site, group in data.groupby("Site"):
                site_corr = group["Dry_Bulb_Temperature_C"].rolling(
                    window=window, min_periods=10
                ).corr(group["Building_Power_kW"])
                correlations.append(site_corr)
            data["temp_power_correlation"] = pd.concat(correlations)
            
            # Is power increasing with temperature? (cooling-dominated)
            data["is_cooling_dominated"] = (data["temp_power_correlation"] > 0).astype(int)
        
        logger.info("Created energy pattern features")
        return data

    def create_site_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create site-based encoding features

        Args:
            data: DataFrame with 'Site' column

        Returns:
            DataFrame with site encoding added
        """
        data = data.copy()

        # One-hot encoding for sites
        site_dummies = pd.get_dummies(data["Site"], prefix="site")
        data = pd.concat([data, site_dummies], axis=1)

        logger.info("Created site encoding features")

        return data

    def create_all_features(
        self,
        data: pd.DataFrame,
        include_lags: bool = True,
        include_rolling: bool = True,
        lag_columns: List[str] = None,
        rolling_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Create all features at once

        Args:
            data: DataFrame to create features from
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling features
            lag_columns: Columns to create lags for
            rolling_columns: Columns to create rolling features for

        Returns:
            DataFrame with all features
        """
        logger.info("Creating features...")

        # Temporal features
        data = self.create_temporal_features(data)

        # Interaction features (must come before energy patterns as it creates power_rate_change)
        data = self.create_interaction_features(data)
        
        # Energy pattern features (uses features from interaction step)
        data = self.create_energy_pattern_features(data)

        # Site encoding
        data = self.create_site_encoding(data)

        # Default columns for lag and rolling features
        if lag_columns is None:
            lag_columns = [
                "Building_Power_kW",
                "Dry_Bulb_Temperature_C",
                "Global_Horizontal_Radiation_W/m2",
            ]

        if rolling_columns is None:
            rolling_columns = lag_columns

        # Lag features
        if include_lags:
            data = self.create_lag_features(data, lag_columns)

        # Rolling features
        if include_rolling:
            data = self.create_rolling_features(data, rolling_columns)

        # Fill any NaN values created by lag/rolling operations
        data = data.fillna(method="bfill").fillna(0)
        
        # Replace inf values with 0 (they typically come from divisions by zero)
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_counts > 0:
            logger.warning(f"Found {inf_counts} inf values, replacing with 0")
        data = data.replace([np.inf, -np.inf], 0)
        
        # Final verification
        nan_counts = data.select_dtypes(include=[np.number]).isna().sum().sum()
        if nan_counts > 0:
            logger.warning(f"Found {nan_counts} NaN values after fillna, replacing with 0")
            data = data.fillna(0)

        # Get feature names (exclude target and identifier columns)
        exclude_cols = [
            "Site",
            "Timestamp_Local",
            "Demand_Response_Flag",
            "Demand_Response_Capacity_kW",
        ]
        self.feature_names = [col for col in data.columns if col not in exclude_cols]

        logger.info(f"Total features created: {len(self.feature_names)}")

        return data

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names

        Returns:
            List of feature column names
        """
        return self.feature_names


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader

    # Load data
    loader = DataLoader("../data")
    train_data = loader.load_training_data(version="v0.2")

    # Create features
    engineer = FeatureEngineer()
    train_data_features = engineer.create_all_features(train_data)

    print(f"\nFeature names: {engineer.get_feature_names()[:10]}...")
    print(f"Data shape: {train_data_features.shape}")
