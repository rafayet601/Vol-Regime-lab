"""Feature engineering for rolling volatility and additional features."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for regime detection models."""
    
    def __init__(self, returns_column: str = "returns"):
        """Initialize feature engineer.
        
        Args:
            returns_column: Name of the returns column
        """
        self.returns_column = returns_column
    
    def compute_rolling_volatility(
        self,
        data: DataFrame,
        window: int = 20,
        method: str = "std",
        annualize: bool = True
    ) -> DataFrame:
        """Compute rolling volatility from returns.
        
        Args:
            data: DataFrame with returns data
            window: Rolling window size
            method: Volatility method ('std', 'ewm', 'garch')
            annualize: Whether to annualize the volatility
            
        Returns:
            DataFrame with rolling volatility added
            
        Raises:
            KeyError: If returns column doesn't exist
            ValueError: If method is not supported
        """
        if self.returns_column not in data.columns:
            raise KeyError(f"Returns column '{self.returns_column}' not found in data")
        
        result = data.copy()
        
        if method == "std":
            # Standard rolling standard deviation
            rolling_std = result[self.returns_column].rolling(window=window).std()
        elif method == "ewm":
            # Exponentially weighted moving standard deviation
            rolling_std = result[self.returns_column].ewm(span=window).std()
        else:
            raise ValueError(f"Unsupported volatility method: {method}")
        
        # Annualize if requested (assuming daily data)
        if annualize:
            rolling_std = rolling_std * np.sqrt(252)
        
        result["rolling_std"] = rolling_std
        
        logger.info(f"Computed rolling volatility with window={window}, method={method}")
        return result
    
    def add_absolute_returns(self, data: DataFrame) -> DataFrame:
        """Add absolute returns feature.
        
        Args:
            data: DataFrame with returns data
            
        Returns:
            DataFrame with absolute returns added
        """
        result = data.copy()
        result["abs_returns"] = np.abs(result[self.returns_column])
        
        logger.debug("Added absolute returns feature")
        return result
    
    def add_negative_returns(self, data: DataFrame) -> DataFrame:
        """Add negative returns indicator feature.
        
        Args:
            data: DataFrame with returns data
            
        Returns:
            DataFrame with negative returns indicator added
        """
        result = data.copy()
        result["negative_returns"] = (result[self.returns_column] < 0).astype(int)
        
        logger.debug("Added negative returns indicator feature")
        return result
    
    def add_z_score_returns(
        self, 
        data: DataFrame, 
        window: int = 252,
        method: str = "rolling"
    ) -> DataFrame:
        """Add z-score normalized returns.
        
        Args:
            data: DataFrame with returns data
            window: Window size for normalization
            method: Normalization method ('rolling' or 'expanding')
            
        Returns:
            DataFrame with z-score returns added
            
        Raises:
            ValueError: If method is not supported
        """
        result = data.copy()
        
        if method == "rolling":
            rolling_mean = result[self.returns_column].rolling(window=window).mean()
            rolling_std = result[self.returns_column].rolling(window=window).std()
        elif method == "expanding":
            rolling_mean = result[self.returns_column].expanding().mean()
            rolling_std = result[self.returns_column].expanding().std()
        else:
            raise ValueError(f"Unsupported z-score method: {method}")
        
        result["z_score_returns"] = (result[self.returns_column] - rolling_mean) / rolling_std
        
        logger.info(f"Added z-score returns with window={window}, method={method}")
        return result
    
    def add_volatility_regime_indicator(
        self,
        data: DataFrame,
        volatility_column: str = "rolling_std",
        quantile_threshold: float = 0.8
    ) -> DataFrame:
        """Add high volatility regime indicator.
        
        Args:
            data: DataFrame with volatility data
            volatility_column: Name of the volatility column
            quantile_threshold: Quantile threshold for high volatility
            
        Returns:
            DataFrame with volatility regime indicator added
            
        Raises:
            KeyError: If volatility column doesn't exist
        """
        if volatility_column not in data.columns:
            raise KeyError(f"Volatility column '{volatility_column}' not found in data")
        
        result = data.copy()
        threshold = result[volatility_column].quantile(quantile_threshold)
        result["high_vol_regime"] = (result[volatility_column] > threshold).astype(int)
        
        logger.info(f"Added volatility regime indicator with threshold={threshold:.4f}")
        return result
    
    def engineer_features(
        self,
        data: DataFrame,
        rolling_window: int = 20,
        additional_features: Optional[List[str]] = None,
        volatility_method: str = "std",
        annualize_vol: bool = True
    ) -> DataFrame:
        """Engineer all features for the model.
        
        Args:
            data: DataFrame with returns data
            rolling_window: Window size for rolling calculations
            additional_features: List of additional features to add
            volatility_method: Method for volatility calculation
            annualize_vol: Whether to annualize volatility
            
        Returns:
            DataFrame with all engineered features
        """
        if additional_features is None:
            additional_features = ["abs_returns", "negative_returns", "z_score_returns"]
        
        # Start with rolling volatility (always needed)
        result = self.compute_rolling_volatility(
            data, 
            window=rolling_window, 
            method=volatility_method,
            annualize=annualize_vol
        )
        
        # Add additional features
        if "abs_returns" in additional_features:
            result = self.add_absolute_returns(result)
        
        if "negative_returns" in additional_features:
            result = self.add_negative_returns(result)
        
        if "z_score_returns" in additional_features:
            result = self.add_z_score_returns(result)
        
        if "high_vol_regime" in additional_features:
            result = self.add_volatility_regime_indicator(result)
        
        # Remove rows with NaN values
        initial_length = len(result)
        result = result.dropna()
        final_length = len(result)
        
        if initial_length != final_length:
            logger.info(f"Removed {initial_length - final_length} rows with NaN values")
        
        logger.info(f"Feature engineering complete: {final_length} observations, "
                   f"{len(result.columns)} features")
        
        return result
    
    def get_feature_names(self, data: DataFrame) -> List[str]:
        """Get list of feature column names (excluding returns and date columns).
        
        Args:
            data: DataFrame with features
            
        Returns:
            List of feature column names
        """
        # Exclude returns column and common non-feature columns
        exclude_columns = {
            self.returns_column, "date", "datetime", "index",
            "open", "high", "low", "close", "volume", "adj close"
        }
        
        feature_columns = [col for col in data.columns if col.lower() not in exclude_columns]
        
        logger.debug(f"Identified {len(feature_columns)} feature columns: {feature_columns}")
        return feature_columns
    
    def validate_features(self, data: DataFrame, feature_columns: List[str]) -> bool:
        """Validate that features are properly engineered.
        
        Args:
            data: DataFrame with features
            feature_columns: List of feature column names
            
        Returns:
            True if features are valid, False otherwise
        """
        # Check that all feature columns exist
        missing_columns = set(feature_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing feature columns: {missing_columns}")
            return False
        
        # Check for infinite values
        for col in feature_columns:
            if np.isinf(data[col]).any():
                logger.error(f"Column '{col}' contains infinite values")
                return False
        
        # Check for excessive NaN values (>50%)
        for col in feature_columns:
            nan_ratio = data[col].isna().sum() / len(data)
            if nan_ratio > 0.5:
                logger.error(f"Column '{col}' has {nan_ratio:.2%} NaN values")
                return False
        
        logger.info("Feature validation passed")
        return True


# Convenience function for quick feature engineering
def engineer_spx_features(
    data: DataFrame,
    rolling_window: int = 20,
    additional_features: Optional[List[str]] = None,
    returns_column: str = "returns"
) -> DataFrame:
    """Convenience function to engineer features for S&P 500 data.
    
    Args:
        data: DataFrame with returns data
        rolling_window: Window size for rolling calculations
        additional_features: List of additional features to add
        returns_column: Name of the returns column
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(returns_column)
    return engineer.engineer_features(
        data, 
        rolling_window=rolling_window, 
        additional_features=additional_features
    )
