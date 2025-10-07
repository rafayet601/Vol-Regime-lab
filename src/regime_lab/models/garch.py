"""GJR-GARCH helper functions via arch package."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from pandas import DataFrame

logger = logging.getLogger(__name__)


class GARCHHelper:
    """Helper class for GARCH model estimation and volatility forecasting."""
    
    def __init__(self, returns_column: str = "returns"):
        """Initialize GARCH helper.
        
        Args:
            returns_column: Name of the returns column
        """
        self.returns_column = returns_column
        self.model = None
        self.is_fitted = False
    
    def fit_garch(
        self,
        data: DataFrame,
        model_type: str = "GARCH",
        vol: str = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        rescale: bool = False
    ) -> Dict:
        """Fit GARCH model to returns data.
        
        Args:
            data: DataFrame with returns data
            model_type: Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')
            vol: Volatility model type
            p: Number of lagged variance terms
            q: Number of lagged error terms
            dist: Distribution for innovations
            rescale: Whether to rescale the data
            
        Returns:
            Dictionary with model results and statistics
            
        Raises:
            KeyError: If returns column doesn't exist
            ValueError: If model fitting fails
        """
        if self.returns_column not in data.columns:
            raise KeyError(f"Returns column '{self.returns_column}' not found in data")
        
        returns = data[self.returns_column].dropna()
        
        if len(returns) < 100:
            raise ValueError("Insufficient data for GARCH estimation (need at least 100 observations)")
        
        logger.info(f"Fitting {model_type} model with {len(returns)} observations")
        
        try:
            # Create GARCH model
            self.model = arch_model(
                returns * 100,  # Scale for numerical stability
                vol=vol,
                p=p,
                q=q,
                dist=dist,
                rescale=rescale
            )
            
            # Fit the model
            results = self.model.fit(disp='off')
            self.is_fitted = True
            
            # Extract key statistics
            model_stats = {
                "aic": results.aic,
                "bic": results.bic,
                "log_likelihood": results.loglikelihood,
                "parameters": results.params.to_dict(),
                "pvalues": results.pvalues.to_dict(),
                "conditional_volatility": results.conditional_volatility / 100,  # Rescale back
                "standardized_residuals": results.std_resid
            }
            
            logger.info(f"GARCH model fitted successfully. AIC: {results.aic:.4f}, BIC: {results.bic:.4f}")
            
            return model_stats
            
        except Exception as e:
            logger.error(f"GARCH model fitting failed: {e}")
            raise ValueError(f"GARCH model fitting failed: {e}")
    
    def forecast_volatility(
        self,
        horizon: int = 1,
        method: str = "simulation",
        simulations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast volatility using fitted GARCH model.
        
        Args:
            horizon: Forecast horizon
            method: Forecasting method
            simulations: Number of simulations for Monte Carlo method
            
        Returns:
            Tuple of (forecast, forecast_variance)
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            forecasts = self.model.forecast(horizon=horizon, method=method, simulations=simulations)
            
            # Extract forecast and variance
            forecast = forecasts.variance.iloc[-horizon:].values.flatten() / 10000  # Rescale
            forecast_var = forecasts.variance.iloc[-horizon:].values.flatten() / 10000
            
            logger.info(f"Generated {horizon}-step volatility forecast")
            
            return forecast, forecast_var
            
        except Exception as e:
            logger.error(f"Volatility forecasting failed: {e}")
            raise
    
    def get_volatility_regime_indicator(
        self,
        data: DataFrame,
        threshold_percentile: float = 75.0
    ) -> DataFrame:
        """Create volatility regime indicator based on GARCH conditional volatility.
        
        Args:
            data: DataFrame with returns data
            threshold_percentile: Percentile threshold for high volatility regime
            
        Returns:
            DataFrame with volatility regime indicator added
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before creating regime indicator")
        
        # Get conditional volatility from fitted model
        cond_vol = self.model.results.conditional_volatility / 100  # Rescale
        
        # Create regime indicator
        threshold = np.percentile(cond_vol, threshold_percentile)
        regime_indicator = (cond_vol > threshold).astype(int)
        
        # Add to original data
        result = data.copy()
        result["garch_volatility"] = cond_vol
        result["garch_high_vol_regime"] = regime_indicator
        
        logger.info(f"Created volatility regime indicator with threshold: {threshold:.4f}")
        
        return result
    
    def compare_with_rolling_volatility(
        self,
        data: DataFrame,
        rolling_window: int = 20
    ) -> DataFrame:
        """Compare GARCH volatility with rolling volatility.
        
        Args:
            data: DataFrame with returns data
            rolling_window: Window size for rolling volatility
            
        Returns:
            DataFrame with both volatility measures
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before comparison")
        
        result = data.copy()
        
        # Add rolling volatility
        result["rolling_vol"] = result[self.returns_column].rolling(window=rolling_window).std()
        
        # Add GARCH conditional volatility
        result["garch_vol"] = self.model.results.conditional_volatility / 100
        
        # Calculate correlation
        valid_data = result[["rolling_vol", "garch_vol"]].dropna()
        if len(valid_data) > 0:
            correlation = valid_data["rolling_vol"].corr(valid_data["garch_vol"])
            logger.info(f"Correlation between rolling and GARCH volatility: {correlation:.4f}")
        
        return result


def fit_garch_to_returns(
    returns: np.ndarray,
    model_type: str = "GARCH",
    vol: str = "GARCH",
    p: int = 1,
    q: int = 1
) -> Dict:
    """Convenience function to fit GARCH model to returns array.
    
    Args:
        returns: Array of returns
        model_type: Type of GARCH model
        vol: Volatility model type
        p: Number of lagged variance terms
        q: Number of lagged error terms
        
    Returns:
        Dictionary with model results
    """
    # Convert to DataFrame
    data = pd.DataFrame({"returns": returns})
    
    # Create helper and fit model
    helper = GARCHHelper()
    return helper.fit_garch(data, model_type=model_type, vol=vol, p=p, q=q)


def create_volatility_features(
    data: DataFrame,
    returns_column: str = "returns",
    garch_params: Optional[Dict] = None,
    rolling_window: int = 20
) -> DataFrame:
    """Create comprehensive volatility features using both GARCH and rolling methods.
    
    Args:
        data: DataFrame with returns data
        returns_column: Name of the returns column
        garch_params: Parameters for GARCH model fitting
        rolling_window: Window size for rolling volatility
        
    Returns:
        DataFrame with volatility features added
    """
    if garch_params is None:
        garch_params = {"model_type": "GARCH", "p": 1, "q": 1}
    
    result = data.copy()
    
    # Add rolling volatility
    result["rolling_vol"] = result[returns_column].rolling(window=rolling_window).std()
    
    try:
        # Try to fit GARCH model
        helper = GARCHHelper(returns_column)
        garch_stats = helper.fit_garch(result, **garch_params)
        
        # Add GARCH volatility
        result["garch_vol"] = garch_stats["conditional_volatility"]
        
        logger.info("Successfully added GARCH volatility features")
        
    except Exception as e:
        logger.warning(f"GARCH fitting failed, using rolling volatility only: {e}")
        result["garch_vol"] = result["rolling_vol"]  # Fallback
    
    return result
