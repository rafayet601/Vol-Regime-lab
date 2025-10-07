"""Regime duration statistics and model diagnostics."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats

logger = logging.getLogger(__name__)


class RegimeDiagnostics:
    """Diagnostics and statistics for regime detection results."""
    
    def __init__(self, regime_column: str = "predicted_state"):
        """Initialize regime diagnostics.
        
        Args:
            regime_column: Name of the regime column
        """
        self.regime_column = regime_column
    
    def compute_regime_durations(
        self, 
        regime_data: DataFrame,
        regime_column: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Compute regime duration statistics.
        
        Args:
            regime_data: DataFrame with regime predictions
            regime_column: Name of the regime column (optional override)
            
        Returns:
            Dictionary with duration statistics for each regime
        """
        if regime_column is None:
            regime_column = self.regime_column
        
        if regime_column not in regime_data.columns:
            raise KeyError(f"Regime column '{regime_column}' not found in data")
        
        regime_series = regime_data[regime_column]
        
        # Find regime changes
        regime_changes = np.where(np.diff(regime_series) != 0)[0]
        
        # Compute durations for each regime
        durations = {0: [], 1: []}  # Assuming 2 states
        
        start_idx = 0
        for change_idx in regime_changes:
            current_regime = regime_series.iloc[start_idx]
            duration = change_idx - start_idx + 1
            durations[current_regime].append(duration)
            start_idx = change_idx + 1
        
        # Add final regime duration
        if start_idx < len(regime_series):
            final_regime = regime_series.iloc[start_idx]
            final_duration = len(regime_series) - start_idx
            durations[final_regime].append(final_duration)
        
        # Compute statistics for each regime
        regime_stats = {}
        for regime, regime_durations in durations.items():
            if regime_durations:
                regime_stats[f"regime_{regime}"] = {
                    "count": len(regime_durations),
                    "mean_duration": np.mean(regime_durations),
                    "median_duration": np.median(regime_durations),
                    "std_duration": np.std(regime_durations),
                    "min_duration": np.min(regime_durations),
                    "max_duration": np.max(regime_durations),
                    "total_periods": np.sum(regime_durations),
                    "durations": regime_durations
                }
            else:
                regime_stats[f"regime_{regime}"] = {
                    "count": 0,
                    "mean_duration": 0,
                    "median_duration": 0,
                    "std_duration": 0,
                    "min_duration": 0,
                    "max_duration": 0,
                    "total_periods": 0,
                    "durations": []
                }
        
        logger.info(f"Computed duration statistics for {len(durations[0]) + len(durations[1])} regime periods")
        
        return regime_stats
    
    def test_regime_persistence(
        self, 
        regime_data: DataFrame,
        regime_column: Optional[str] = None
    ) -> Dict[str, float]:
        """Test for regime persistence (duration distribution tests).
        
        Args:
            regime_data: DataFrame with regime predictions
            regime_column: Name of the regime column
            
        Returns:
            Dictionary with persistence test results
        """
        regime_stats = self.compute_regime_durations(regime_data, regime_column)
        
        persistence_results = {}
        
        for regime_key, stats in regime_stats.items():
            durations = stats["durations"]
            
            if len(durations) > 3:  # Need sufficient data for tests
                # Test against exponential distribution (memoryless property)
                # If regimes are memoryless, durations should follow exponential distribution
                try:
                    # Fit exponential distribution
                    lambda_exp = 1 / np.mean(durations)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.kstest(
                        durations, 
                        lambda x: stats.expon.cdf(x, scale=1/lambda_exp)
                    )
                    
                    # Anderson-Darling test
                    ad_stat, ad_critical, ad_significance = stats.anderson(
                        durations, dist='expon'
                    )
                    
                    persistence_results[regime_key] = {
                        "ks_statistic": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "ad_statistic": ad_stat,
                        "ad_critical": ad_critical[2],  # 5% significance level
                        "exponential_lambda": lambda_exp,
                        "is_memoryless": ks_pvalue > 0.05  # Memoryless if exponential
                    }
                    
                except Exception as e:
                    logger.warning(f"Persistence test failed for {regime_key}: {e}")
                    persistence_results[regime_key] = {
                        "error": str(e),
                        "is_memoryless": None
                    }
            else:
                persistence_results[regime_key] = {
                    "error": "Insufficient data for persistence test",
                    "is_memoryless": None
                }
        
        return persistence_results
    
    def compute_transition_statistics(
        self, 
        regime_data: DataFrame,
        regime_column: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute transition matrix and related statistics.
        
        Args:
            regime_data: DataFrame with regime predictions
            regime_column: Name of the regime column
            
        Returns:
            Dictionary with transition statistics
        """
        if regime_column is None:
            regime_column = self.regime_column
        
        regime_series = regime_data[regime_column]
        
        # Compute transition counts
        n_states = len(regime_series.unique())
        transition_counts = np.zeros((n_states, n_states))
        
        for i in range(len(regime_series) - 1):
            current_state = int(regime_series.iloc[i])
            next_state = int(regime_series.iloc[i + 1])
            transition_counts[current_state, next_state] += 1
        
        # Compute transition probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_probs = transition_counts / row_sums[:, np.newaxis]
        
        # Compute additional statistics
        total_transitions = np.sum(transition_counts) - np.sum(np.diag(transition_counts))
        total_periods = len(regime_series) - 1
        transition_rate = total_transitions / total_periods
        
        # Average regime duration (inverse of transition rate)
        avg_duration = 1 / transition_rate if transition_rate > 0 else np.inf
        
        results = {
            "transition_counts": transition_counts.tolist(),
            "transition_probabilities": transition_probs.tolist(),
            "total_transitions": total_transitions,
            "total_periods": total_periods,
            "transition_rate": transition_rate,
            "average_duration": avg_duration,
            "diagonal_probability": np.mean(np.diag(transition_probs))  # Persistence measure
        }
        
        logger.info(f"Computed transition statistics: {total_transitions} transitions, "
                   f"rate={transition_rate:.4f}")
        
        return results
    
    def analyze_regime_characteristics(
        self,
        regime_data: DataFrame,
        feature_data: Optional[DataFrame] = None,
        regime_column: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Analyze characteristics of each regime.
        
        Args:
            regime_data: DataFrame with regime predictions
            feature_data: Optional DataFrame with features for analysis
            regime_column: Name of the regime column
            
        Returns:
            Dictionary with regime characteristics
        """
        if regime_column is None:
            regime_column = self.regime_column
        
        regime_series = regime_data[regime_column]
        unique_regimes = regime_series.unique()
        
        regime_characteristics = {}
        
        for regime in unique_regimes:
            regime_mask = regime_series == regime
            regime_periods = regime_mask.sum()
            
            characteristics = {
                "frequency": regime_periods,
                "percentage": regime_periods / len(regime_series) * 100,
                "mean_duration": regime_periods / (regime_mask.diff().abs().sum() + 1)
            }
            
            # Add feature analysis if feature data is provided
            if feature_data is not None:
                # Use iloc with boolean array converted to indices
                regime_indices = regime_mask[regime_mask].index
                regime_features = feature_data.loc[regime_indices]
                
                feature_stats = {}
                for col in regime_features.columns:
                    if regime_features[col].dtype in ['float64', 'int64']:
                        feature_stats[col] = {
                            "mean": regime_features[col].mean(),
                            "std": regime_features[col].std(),
                            "min": regime_features[col].min(),
                            "max": regime_features[col].max(),
                            "median": regime_features[col].median()
                        }
                
                characteristics["feature_statistics"] = feature_stats
            
            regime_characteristics[f"regime_{int(regime)}"] = characteristics
        
        logger.info(f"Analyzed characteristics for {len(unique_regimes)} regimes")
        
        return regime_characteristics
    
    def generate_diagnostic_report(
        self,
        regime_data: DataFrame,
        feature_data: Optional[DataFrame] = None,
        regime_column: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Generate comprehensive diagnostic report.
        
        Args:
            regime_data: DataFrame with regime predictions
            feature_data: Optional DataFrame with features
            regime_column: Name of the regime column
            
        Returns:
            Dictionary with comprehensive diagnostic results
        """
        logger.info("Generating comprehensive diagnostic report...")
        
        # Compute all diagnostics
        duration_stats = self.compute_regime_durations(regime_data, regime_column)
        persistence_results = self.test_regime_persistence(regime_data, regime_column)
        transition_stats = self.compute_transition_statistics(regime_data, regime_column)
        regime_characteristics = self.analyze_regime_characteristics(
            regime_data, feature_data, regime_column
        )
        
        # Compile comprehensive report
        diagnostic_report = {
            "duration_statistics": duration_stats,
            "persistence_tests": persistence_results,
            "transition_statistics": transition_stats,
            "regime_characteristics": regime_characteristics,
            "summary": {
                "total_observations": len(regime_data),
                "number_of_regimes": len(regime_data[regime_column or self.regime_column].unique()),
                "total_transitions": transition_stats["total_transitions"],
                "average_regime_duration": transition_stats["average_duration"],
                "regime_persistence": transition_stats["diagonal_probability"]
            }
        }
        
        logger.info("Diagnostic report generated successfully")
        
        return diagnostic_report
    
    def print_diagnostic_summary(self, diagnostic_report: Dict[str, Dict]) -> None:
        """Print a formatted diagnostic summary.
        
        Args:
            diagnostic_report: Diagnostic report dictionary
        """
        summary = diagnostic_report["summary"]
        
        print("\n" + "="*60)
        print("REGIME DIAGNOSTIC SUMMARY")
        print("="*60)
        print(f"Total Observations: {summary['total_observations']}")
        print(f"Number of Regimes: {summary['number_of_regimes']}")
        print(f"Total Transitions: {summary['total_transitions']}")
        print(f"Average Regime Duration: {summary['average_regime_duration']:.1f} periods")
        print(f"Regime Persistence: {summary['regime_persistence']:.3f}")
        
        # Duration statistics
        print("\nDuration Statistics:")
        duration_stats = diagnostic_report["duration_statistics"]
        for regime_key, stats in duration_stats.items():
            if stats["count"] > 0:
                regime_name = "Low Volatility" if "0" in regime_key else "High Volatility"
                print(f"  {regime_name}:")
                print(f"    Count: {stats['count']} periods")
                print(f"    Mean Duration: {stats['mean_duration']:.1f} periods")
                print(f"    Median Duration: {stats['median_duration']:.1f} periods")
                print(f"    Total Periods: {stats['total_periods']}")
        
        # Persistence tests
        print("\nPersistence Tests:")
        persistence_results = diagnostic_report["persistence_tests"]
        for regime_key, results in persistence_results.items():
            if "is_memoryless" in results and results["is_memoryless"] is not None:
                regime_name = "Low Volatility" if "0" in regime_key else "High Volatility"
                memoryless = "Yes" if results["is_memoryless"] else "No"
                print(f"  {regime_name}: Memoryless = {memoryless} (p={results['ks_pvalue']:.3f})")
        
        print("="*60)


# Convenience functions
def analyze_regimes_from_artifacts(
    artifacts_dir: str = "artifacts",
    regime_column: str = "predicted_state"
) -> Dict[str, Dict]:
    """Analyze regimes from saved artifacts.
    
    Args:
        artifacts_dir: Directory containing artifacts
        regime_column: Name of the regime column
        
    Returns:
        Diagnostic report
    """
    from ..utils.io import load_dataframe
    
    # Load predictions
    predictions = load_dataframe(f"{artifacts_dir}/posteriors.csv")
    
    # Load features if available
    try:
        features = load_dataframe(f"{artifacts_dir}/features.csv")
    except FileNotFoundError:
        features = None
        logger.warning("Features file not found, skipping feature analysis")
    
    # Create diagnostics
    diagnostics = RegimeDiagnostics(regime_column)
    return diagnostics.generate_diagnostic_report(predictions, features, regime_column)


def quick_regime_summary(regime_data: DataFrame, regime_column: str = "predicted_state") -> None:
    """Print quick regime summary.
    
    Args:
        regime_data: DataFrame with regime predictions
        regime_column: Name of the regime column
    """
    diagnostics = RegimeDiagnostics(regime_column)
    report = diagnostics.generate_diagnostic_report(regime_data)
    diagnostics.print_diagnostic_summary(report)
