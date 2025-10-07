"""S&P 500 data loader with returns computation using yfinance."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas import DataFrame

from ..utils.config import ensure_dir
from ..utils.io import file_exists, load_pickle, save_pickle

logger = logging.getLogger(__name__)


class SPXDataLoader:
    """S&P 500 data loader with caching and returns computation."""
    
    def __init__(self, cache_dir: str = "./data/raw"):
        """Initialize the data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        ensure_dir(str(self.cache_dir))
        
    def load_data(
        self,
        symbol: str = "^GSPC",
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> DataFrame:
        """Load S&P 500 price data with caching.
        
        Args:
            symbol: Stock symbol (default: ^GSPC for S&P 500)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data and datetime index
            
        Raises:
            ValueError: If date format is invalid
            RuntimeError: If data download fails
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create cache filename
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.pkl"
        
        # Try to load from cache first
        if use_cache and file_exists(str(cache_file)):
            logger.info(f"Loading cached data from {cache_file}")
            return load_pickle(str(cache_file))
        
        # Download fresh data
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise RuntimeError(f"No data retrieved for {symbol}")
            
            # Clean up column names
            data.columns = [col.lower() for col in data.columns]
            
            # Cache the data
            save_pickle(data, str(cache_file))
            logger.info(f"Data cached to {cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise RuntimeError(f"Data download failed: {e}")
    
    def compute_returns(
        self,
        price_data: DataFrame,
        price_column: str = "close",
        method: str = "log",
        periods: int = 1
    ) -> DataFrame:
        """Compute returns from price data.
        
        Args:
            price_data: DataFrame with price data
            price_column: Column name containing prices
            method: Return calculation method ('log' or 'simple')
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame with returns added
            
        Raises:
            KeyError: If price column doesn't exist
            ValueError: If method is not supported
        """
        if price_column not in price_data.columns:
            raise KeyError(f"Price column '{price_column}' not found in data")
        
        data = price_data.copy()
        
        if method == "log":
            data["returns"] = data[price_column].pct_change(periods=periods)
            # Convert to log returns
            data["returns"] = data["returns"].apply(lambda x: np.log(1 + x) if pd.notna(x) else x)
        elif method == "simple":
            data["returns"] = data[price_column].pct_change(periods=periods)
        else:
            raise ValueError(f"Unsupported return method: {method}")
        
        # Remove the first row with NaN returns
        data = data.dropna(subset=["returns"])
        
        logger.info(f"Computed {method} returns for {len(data)} observations")
        return data
    
    def get_full_dataset(
        self,
        symbol: str = "^GSPC",
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        price_column: str = "close",
        return_method: str = "log"
    ) -> Tuple[DataFrame, DataFrame]:
        """Get complete dataset with prices and returns.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            price_column: Price column for returns calculation
            return_method: Method for returns calculation
            
        Returns:
            Tuple of (price_data, returns_data)
        """
        # Load price data
        price_data = self.load_data(symbol, start_date, end_date)
        
        # Compute returns
        returns_data = self.compute_returns(
            price_data, 
            price_column=price_column, 
            method=return_method
        )
        
        logger.info(f"Dataset loaded: {len(price_data)} price observations, "
                   f"{len(returns_data)} return observations")
        
        return price_data, returns_data
    
    def validate_data(self, data: DataFrame, required_columns: list) -> bool:
        """Validate that data contains required columns and has no critical issues.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check required columns
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for empty data
        if data.empty:
            logger.error("Data is empty")
            return False
        
        # Check for all NaN values in required columns
        for col in required_columns:
            if data[col].isna().all():
                logger.error(f"Column '{col}' contains only NaN values")
                return False
        
        logger.info("Data validation passed")
        return True


# Convenience function for quick data loading
def load_spx_data(
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "./data/raw"
) -> Tuple[DataFrame, DataFrame]:
    """Convenience function to load S&P 500 data with returns.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        cache_dir: Cache directory
        
    Returns:
        Tuple of (price_data, returns_data)
    """
    loader = SPXDataLoader(cache_dir)
    return loader.get_full_dataset(start_date=start_date, end_date=end_date)
