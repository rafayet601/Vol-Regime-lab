"""Tests for feature engineering functionality."""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.data.features import FeatureEngineer, engineer_spx_features


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        
        return pd.DataFrame({
            'returns': returns
        }, index=dates)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer(returns_column="returns")
    
    def test_rolling_volatility_std(self, feature_engineer, sample_returns_data):
        """Test rolling volatility calculation with standard deviation."""
        result = feature_engineer.compute_rolling_volatility(
            sample_returns_data, window=20, method="std"
        )
        
        assert "rolling_std" in result.columns
        assert len(result) == len(sample_returns_data)
        assert not result["rolling_std"].isna().all()
        
        # First 19 values should be NaN (window size - 1)
        assert result["rolling_std"].iloc[:19].isna().all()
        assert not result["rolling_std"].iloc[19:].isna().all()
    
    def test_rolling_volatility_ewm(self, feature_engineer, sample_returns_data):
        """Test rolling volatility calculation with exponential weighting."""
        result = feature_engineer.compute_rolling_volatility(
            sample_returns_data, window=20, method="ewm"
        )
        
        assert "rolling_std" in result.columns
        assert len(result) == len(sample_returns_data)
        assert not result["rolling_std"].isna().all()
    
    def test_rolling_volatility_annualized(self, feature_engineer, sample_returns_data):
        """Test annualized rolling volatility."""
        result_annualized = feature_engineer.compute_rolling_volatility(
            sample_returns_data, window=20, annualize=True
        )
        
        result_daily = feature_engineer.compute_rolling_volatility(
            sample_returns_data, window=20, annualize=False
        )
        
        # Annualized should be larger (multiplied by sqrt(252))
        annualized_factor = np.sqrt(252)
        assert np.allclose(
            result_annualized["rolling_std"].iloc[19:],
            result_daily["rolling_std"].iloc[19:] * annualized_factor,
            rtol=1e-10
        )
    
    def test_rolling_volatility_invalid_method(self, feature_engineer, sample_returns_data):
        """Test rolling volatility with invalid method."""
        with pytest.raises(ValueError, match="Unsupported volatility method"):
            feature_engineer.compute_rolling_volatility(
                sample_returns_data, window=20, method="invalid"
            )
    
    def test_rolling_volatility_missing_column(self, feature_engineer, sample_returns_data):
        """Test rolling volatility with missing returns column."""
        data_no_returns = sample_returns_data.drop(columns=["returns"])
        
        with pytest.raises(KeyError, match="Returns column 'returns' not found"):
            feature_engineer.compute_rolling_volatility(data_no_returns, window=20)
    
    def test_add_absolute_returns(self, feature_engineer, sample_returns_data):
        """Test absolute returns feature."""
        result = feature_engineer.add_absolute_returns(sample_returns_data)
        
        assert "abs_returns" in result.columns
        assert np.allclose(result["abs_returns"], np.abs(result["returns"]))
    
    def test_add_negative_returns(self, feature_engineer, sample_returns_data):
        """Test negative returns indicator."""
        result = feature_engineer.add_negative_returns(sample_returns_data)
        
        assert "negative_returns" in result.columns
        assert result["negative_returns"].dtype == int
        assert np.allclose(
            result["negative_returns"], 
            (result["returns"] < 0).astype(int)
        )
    
    def test_add_z_score_returns_rolling(self, feature_engineer, sample_returns_data):
        """Test z-score returns with rolling window."""
        result = feature_engineer.add_z_score_returns(
            sample_returns_data, window=20, method="rolling"
        )
        
        assert "z_score_returns" in result.columns
        assert len(result) == len(sample_returns_data)
        
        # Z-scores should have mean ~0 and std ~1 for the rolling window
        valid_z_scores = result["z_score_returns"].dropna()
        assert abs(valid_z_scores.mean()) < 0.1  # Close to zero
        assert abs(valid_z_scores.std() - 1) < 0.1  # Close to one
    
    def test_add_z_score_returns_expanding(self, feature_engineer, sample_returns_data):
        """Test z-score returns with expanding window."""
        result = feature_engineer.add_z_score_returns(
            sample_returns_data, window=20, method="expanding"
        )
        
        assert "z_score_returns" in result.columns
        assert len(result) == len(sample_returns_data)
    
    def test_add_z_score_returns_invalid_method(self, feature_engineer, sample_returns_data):
        """Test z-score returns with invalid method."""
        with pytest.raises(ValueError, match="Unsupported z-score method"):
            feature_engineer.add_z_score_returns(
                sample_returns_data, window=20, method="invalid"
            )
    
    def test_add_volatility_regime_indicator(self, feature_engineer, sample_returns_data):
        """Test volatility regime indicator."""
        # First add volatility
        data_with_vol = feature_engineer.compute_rolling_volatility(
            sample_returns_data, window=20
        )
        
        result = feature_engineer.add_volatility_regime_indicator(
            data_with_vol, quantile_threshold=0.8
        )
        
        assert "high_vol_regime" in result.columns
        assert result["high_vol_regime"].dtype == int
        assert set(result["high_vol_regime"].unique()).issubset({0, 1})
        
        # High volatility regime should be approximately 20% of observations
        high_vol_pct = result["high_vol_regime"].mean() * 100
        assert 15 <= high_vol_pct <= 25  # Allow some tolerance
    
    def test_add_volatility_regime_missing_column(self, feature_engineer, sample_returns_data):
        """Test volatility regime indicator with missing volatility column."""
        with pytest.raises(KeyError, match="Volatility column 'rolling_std' not found"):
            feature_engineer.add_volatility_regime_indicator(
                sample_returns_data, volatility_column="rolling_std"
            )
    
    def test_engineer_features_comprehensive(self, feature_engineer, sample_returns_data):
        """Test comprehensive feature engineering."""
        additional_features = ["abs_returns", "negative_returns", "z_score_returns"]
        
        result = feature_engineer.engineer_features(
            sample_returns_data,
            rolling_window=20,
            additional_features=additional_features,
            volatility_method="std",
            annualize_vol=True
        )
        
        expected_columns = ["rolling_std"] + additional_features
        for col in expected_columns:
            assert col in result.columns
        
        # Should remove NaN rows
        assert not result.isna().any().any()
        assert len(result) < len(sample_returns_data)  # Some rows removed due to NaN
    
    def test_engineer_features_with_high_vol_regime(self, feature_engineer, sample_returns_data):
        """Test feature engineering including high volatility regime."""
        additional_features = ["abs_returns", "negative_returns", "high_vol_regime"]
        
        result = feature_engineer.engineer_features(
            sample_returns_data,
            rolling_window=20,
            additional_features=additional_features
        )
        
        for col in additional_features:
            assert col in result.columns
        
        if "high_vol_regime" in result.columns:
            assert set(result["high_vol_regime"].unique()).issubset({0, 1})
    
    def test_get_feature_names(self, feature_engineer, sample_returns_data):
        """Test feature name extraction."""
        # Add some features
        result = feature_engineer.engineer_features(
            sample_returns_data,
            additional_features=["abs_returns", "negative_returns"]
        )
        
        feature_names = feature_engineer.get_feature_names(result)
        
        assert "returns" not in feature_names
        assert "rolling_std" in feature_names
        assert "abs_returns" in feature_names
        assert "negative_returns" in feature_names
    
    def test_validate_features_valid(self, feature_engineer, sample_returns_data):
        """Test feature validation with valid features."""
        result = feature_engineer.engineer_features(
            sample_returns_data,
            additional_features=["abs_returns", "negative_returns"]
        )
        
        feature_names = feature_engineer.get_feature_names(result)
        assert feature_engineer.validate_features(result, feature_names)
    
    def test_validate_features_missing_column(self, feature_engineer, sample_returns_data):
        """Test feature validation with missing column."""
        result = feature_engineer.engineer_features(sample_returns_data)
        feature_names = feature_engineer.get_feature_names(result)
        
        # Add non-existent column to feature names
        feature_names.append("nonexistent_column")
        
        assert not feature_engineer.validate_features(result, feature_names)
    
    def test_validate_features_infinite_values(self, feature_engineer, sample_returns_data):
        """Test feature validation with infinite values."""
        result = feature_engineer.engineer_features(sample_returns_data)
        
        # Introduce infinite values
        result.loc[result.index[0], "rolling_std"] = np.inf
        
        feature_names = feature_engineer.get_feature_names(result)
        assert not feature_engineer.validate_features(result, feature_names)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_engineer_spx_features(self):
        """Test convenience function for S&P 500 feature engineering."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0, 0.02, 100)
        
        data = pd.DataFrame({
            'returns': returns
        }, index=dates)
        
        result = engineer_spx_features(
            data,
            rolling_window=20,
            additional_features=["abs_returns", "negative_returns"]
        )
        
        assert "rolling_std" in result.columns
        assert "abs_returns" in result.columns
        assert "negative_returns" in result.columns
        assert len(result) > 0
    
    def test_engineer_spx_features_custom_returns_column(self):
        """Test convenience function with custom returns column."""
        # Create sample data with custom returns column name
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0, 0.02, 100)
        
        data = pd.DataFrame({
            'log_returns': returns
        }, index=dates)
        
        result = engineer_spx_features(
            data,
            returns_column="log_returns",
            additional_features=["abs_returns"]
        )
        
        assert "rolling_std" in result.columns
        assert "abs_returns" in result.columns
        assert len(result) > 0


class TestFeatureEngineerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        engineer = FeatureEngineer()
        empty_df = pd.DataFrame(columns=['returns'])
        
        with pytest.raises((KeyError, ValueError)):
            engineer.compute_rolling_volatility(empty_df, window=20)
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        engineer = FeatureEngineer()
        single_row_df = pd.DataFrame({'returns': [0.01]})
        
        # Should handle gracefully
        result = engineer.compute_rolling_volatility(single_row_df, window=20)
        assert len(result) == 1
        assert result["rolling_std"].iloc[0] is np.nan
    
    def test_constant_returns(self):
        """Test with constant returns (zero volatility)."""
        engineer = FeatureEngineer()
        constant_returns_df = pd.DataFrame({
            'returns': np.ones(100) * 0.01  # Constant 1% returns
        })
        
        result = engineer.compute_rolling_volatility(constant_returns_df, window=20)
        
        # Rolling std should be zero for constant returns
        valid_std = result["rolling_std"].dropna()
        assert np.allclose(valid_std, 0, atol=1e-10)
    
    def test_large_window_size(self):
        """Test with window size larger than data length."""
        engineer = FeatureEngineer()
        small_data = pd.DataFrame({
            'returns': np.random.normal(0, 0.02, 10)
        })
        
        # Should handle gracefully
        result = engineer.compute_rolling_volatility(small_data, window=50)
        assert len(result) == 10
        assert result["rolling_std"].isna().all()  # All NaN due to large window
