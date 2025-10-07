"""Tests for data loading functionality."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.data.loader import SPXDataLoader, load_spx_data


class TestSPXDataLoader:
    """Test cases for SPXDataLoader class."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100))
        
        return pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    @pytest.fixture
    def mock_yfinance_data(self, sample_price_data):
        """Mock yfinance data."""
        # Convert to yfinance format
        data = sample_price_data.copy()
        data.columns = [col.lower() for col in data.columns]
        return data
    
    @pytest.fixture
    def data_loader(self, tmp_path):
        """Create SPXDataLoader instance with temporary cache directory."""
        return SPXDataLoader(cache_dir=str(tmp_path))
    
    @patch('regime_lab.data.loader.yf.Ticker')
    def test_load_data_success(self, mock_ticker, data_loader, mock_yfinance_data):
        """Test successful data loading."""
        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_yfinance_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Load data
        result = data_loader.load_data(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2020-04-10"
        )
        
        # Verify results
        assert len(result) == 100
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(result.index, pd.DatetimeIndex)
        
        # Verify mock was called correctly
        mock_ticker.assert_called_once_with("^GSPC")
        mock_ticker_instance.history.assert_called_once_with(
            start="2020-01-01", end="2020-04-10"
        )
    
    @patch('regime_lab.data.loader.yf.Ticker')
    def test_load_data_empty_result(self, mock_ticker, data_loader):
        """Test data loading with empty result."""
        # Setup mock to return empty DataFrame
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Should raise RuntimeError for empty data
        with pytest.raises(RuntimeError, match="No data retrieved"):
            data_loader.load_data(
                symbol="^GSPC",
                start_date="2020-01-01",
                end_date="2020-04-10"
            )
    
    @patch('regime_lab.data.loader.yf.Ticker')
    def test_load_data_with_caching(self, mock_ticker, data_loader, mock_yfinance_data, tmp_path):
        """Test data loading with caching."""
        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_yfinance_data
        mock_ticker.return_value = mock_ticker_instance
        
        # First load - should download and cache
        result1 = data_loader.load_data(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2020-04-10",
            use_cache=True
        )
        
        # Second load - should use cache
        result2 = data_loader.load_data(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2020-04-10",
            use_cache=True
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Mock should only be called once (first load)
        assert mock_ticker.call_count == 1
        
        # Cache file should exist
        cache_file = tmp_path / "^GSPC_2020-01-01_2020-04-10.pkl"
        assert cache_file.exists()
    
    def test_load_data_without_caching(self, data_loader, mock_yfinance_data):
        """Test data loading without caching."""
        with patch('regime_lab.data.loader.yf.Ticker') as mock_ticker:
            # Setup mock
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_yfinance_data
            mock_ticker.return_value = mock_ticker_instance
            
            # Load without caching
            result = data_loader.load_data(
                symbol="^GSPC",
                start_date="2020-01-01",
                end_date="2020-04-10",
                use_cache=False
            )
            
            assert len(result) == 100
            mock_ticker.assert_called_once()
    
    def test_compute_returns_simple(self, data_loader, sample_price_data):
        """Test simple returns computation."""
        # Convert to yfinance format
        data = sample_price_data.copy()
        data.columns = [col.lower() for col in data.columns]
        
        result = data_loader.compute_returns(
            data, 
            price_column="close", 
            method="simple"
        )
        
        assert "returns" in result.columns
        assert len(result) == len(data) - 1  # One less due to first NaN
        assert not result["returns"].isna().all()
        
        # Verify returns calculation
        expected_returns = data["close"].pct_change().iloc[1:]
        np.testing.assert_array_almost_equal(
            result["returns"].values, 
            expected_returns.values,
            decimal=10
        )
    
    def test_compute_returns_log(self, data_loader, sample_price_data):
        """Test log returns computation."""
        # Convert to yfinance format
        data = sample_price_data.copy()
        data.columns = [col.lower() for col in data.columns]
        
        result = data_loader.compute_returns(
            data, 
            price_column="close", 
            method="log"
        )
        
        assert "returns" in result.columns
        assert len(result) == len(data) - 1
        
        # Log returns should be smaller than simple returns
        simple_returns = data["close"].pct_change().iloc[1:]
        log_returns = np.log(1 + simple_returns)
        
        np.testing.assert_array_almost_equal(
            result["returns"].values, 
            log_returns.values,
            decimal=10
        )
    
    def test_compute_returns_multiple_periods(self, data_loader, sample_price_data):
        """Test returns computation with multiple periods."""
        # Convert to yfinance format
        data = sample_price_data.copy()
        data.columns = [col.lower() for col in data.columns]
        
        result = data_loader.compute_returns(
            data, 
            price_column="close", 
            method="simple",
            periods=5
        )
        
        assert "returns" in result.columns
        assert len(result) == len(data) - 5  # Five less due to first 5 NaN values
        
        # Verify 5-period returns calculation
        expected_returns = data["close"].pct_change(periods=5).iloc[5:]
        np.testing.assert_array_almost_equal(
            result["returns"].values, 
            expected_returns.values,
            decimal=10
        )
    
    def test_compute_returns_invalid_method(self, data_loader, sample_price_data):
        """Test returns computation with invalid method."""
        # Convert to yfinance format
        data = sample_price_data.copy()
        data.columns = [col.lower() for col in data.columns]
        
        with pytest.raises(ValueError, match="Unsupported return method"):
            data_loader.compute_returns(
                data, 
                price_column="close", 
                method="invalid"
            )
    
    def test_compute_returns_missing_column(self, data_loader, sample_price_data):
        """Test returns computation with missing price column."""
        with pytest.raises(KeyError, match="Price column 'nonexistent' not found"):
            data_loader.compute_returns(
                sample_price_data, 
                price_column="nonexistent"
            )
    
    def test_get_full_dataset(self, data_loader, mock_yfinance_data):
        """Test getting full dataset with prices and returns."""
        with patch('regime_lab.data.loader.yf.Ticker') as mock_ticker:
            # Setup mock
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_yfinance_data
            mock_ticker.return_value = mock_ticker_instance
            
            price_data, returns_data = data_loader.get_full_dataset(
                symbol="^GSPC",
                start_date="2020-01-01",
                end_date="2020-04-10"
            )
            
            # Verify price data
            assert len(price_data) == 100
            assert list(price_data.columns) == ['open', 'high', 'low', 'close', 'volume']
            
            # Verify returns data
            assert len(returns_data) == 99  # One less due to returns calculation
            assert "returns" in returns_data.columns
            assert not returns_data["returns"].isna().all()
    
    def test_validate_data_valid(self, data_loader, sample_price_data):
        """Test data validation with valid data."""
        required_columns = ['open', 'close']
        data = sample_price_data.copy()
        data.columns = [col.lower() for col in data.columns]
        
        assert data_loader.validate_data(data, required_columns)
    
    def test_validate_data_missing_columns(self, data_loader, sample_price_data):
        """Test data validation with missing columns."""
        required_columns = ['open', 'close', 'nonexistent']
        data = sample_price_data.copy()
        data.columns = [col.lower() for col in data.columns]
        
        assert not data_loader.validate_data(data, required_columns)
    
    def test_validate_data_empty(self, data_loader):
        """Test data validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        required_columns = ['open', 'close']
        
        assert not data_loader.validate_data(empty_df, required_columns)
    
    def test_validate_data_all_nan(self, data_loader):
        """Test data validation with all NaN values."""
        data = pd.DataFrame({
            'open': [np.nan] * 10,
            'close': [1.0] * 10
        })
        required_columns = ['open', 'close']
        
        assert not data_loader.validate_data(data, required_columns)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('regime_lab.data.loader.yf.Ticker')
    def test_load_spx_data(self, mock_ticker, tmp_path):
        """Test convenience function for loading S&P 500 data."""
        # Create mock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 100))
        
        mock_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Load data
        price_data, returns_data = load_spx_data(
            start_date="2020-01-01",
            end_date="2020-04-10",
            cache_dir=str(tmp_path)
        )
        
        # Verify results
        assert len(price_data) == 100
        assert len(returns_data) == 99
        assert "returns" in returns_data.columns
        
        # Verify mock was called
        mock_ticker.assert_called_once_with("^GSPC")


class TestSPXDataLoaderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_date_format(self):
        """Test with invalid date format."""
        loader = SPXDataLoader()
        
        # This should not raise an error immediately, but yfinance might handle it
        # We're testing that our code doesn't crash with invalid dates
        with patch('regime_lab.data.loader.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = Exception("Invalid date")
            mock_ticker.return_value = mock_ticker_instance
            
            with pytest.raises(RuntimeError, match="Data download failed"):
                loader.load_data(
                    symbol="^GSPC",
                    start_date="invalid-date",
                    end_date="2020-04-10"
                )
    
    def test_network_error(self):
        """Test handling of network errors."""
        loader = SPXDataLoader()
        
        with patch('regime_lab.data.loader.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = Exception("Network error")
            mock_ticker.return_value = mock_ticker_instance
            
            with pytest.raises(RuntimeError, match="Data download failed"):
                loader.load_data(
                    symbol="^GSPC",
                    start_date="2020-01-01",
                    end_date="2020-04-10"
                )
    
    def test_constant_price_data(self):
        """Test with constant price data."""
        loader = SPXDataLoader()
        
        # Create constant price data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        constant_data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [100.0] * 10,
            'low': [100.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000000] * 10
        }, index=dates)
        
        result = loader.compute_returns(constant_data, method="simple")
        
        # Returns should be zero for constant prices
        assert result["returns"].iloc[1:].sum() == 0.0
    
    def test_single_row_data(self):
        """Test with single row of data."""
        loader = SPXDataLoader()
        
        single_row = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000000]
        })
        
        result = loader.compute_returns(single_row, method="simple")
        
        # Should return empty DataFrame (no returns possible with single observation)
        assert len(result) == 0
    
    def test_cache_directory_creation(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        non_existent_dir = tmp_path / "nonexistent" / "cache"
        
        # Should not raise error, should create directory
        loader = SPXDataLoader(cache_dir=str(non_existent_dir))
        assert non_existent_dir.exists()
