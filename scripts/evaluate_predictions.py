#!/usr/bin/env python3
"""Evaluate regime detection model predictions."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.utils.config import ensure_dir
from regime_lab.utils.io import load_dataframe


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def load_predictions_and_features(artifacts_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and features from artifacts directory."""
    predictions = load_dataframe(f"{artifacts_dir}/posteriors.csv")
    features = load_dataframe(f"{artifacts_dir}/features.csv")
    return predictions, features


def evaluate_regime_separation(predictions: pd.DataFrame, features: pd.DataFrame) -> Dict:
    """Evaluate how well regimes are separated based on features.
    
    This measures if the two regimes have meaningfully different characteristics.
    """
    logger = logging.getLogger(__name__)
    
    # Merge data
    data = pd.merge(predictions, features, left_index=True, right_index=True, how='inner')
    
    # Get volatility measure
    vol_col = 'rolling_std' if 'rolling_std' in data.columns else None
    if vol_col is None:
        logger.warning("No volatility column found for separation analysis")
        return {}
    
    # Compute statistics for each regime
    regime_0_vol = data[data['predicted_state'] == 0][vol_col]
    regime_1_vol = data[data['predicted_state'] == 1][vol_col]
    
    # Statistical tests
    from scipy import stats
    
    # T-test: Are the means significantly different?
    t_stat, t_pvalue = stats.ttest_ind(regime_0_vol, regime_1_vol)
    
    # Effect size (Cohen's d): How large is the difference?
    mean_diff = regime_1_vol.mean() - regime_0_vol.mean()
    pooled_std = np.sqrt((regime_0_vol.std()**2 + regime_1_vol.std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    results = {
        'regime_0_mean_vol': regime_0_vol.mean(),
        'regime_0_std_vol': regime_0_vol.std(),
        'regime_1_mean_vol': regime_1_vol.mean(),
        'regime_1_std_vol': regime_1_vol.std(),
        'mean_difference': mean_diff,
        'relative_difference_pct': (mean_diff / regime_0_vol.mean()) * 100,
        't_statistic': t_stat,
        'p_value': t_pvalue,
        'cohens_d': cohens_d,
        'overlap_coefficient': _compute_overlap(regime_0_vol, regime_1_vol)
    }
    
    return results


def _compute_overlap(regime_0: pd.Series, regime_1: pd.Series) -> float:
    """Compute overlap coefficient between two distributions (0=no overlap, 1=complete overlap)."""
    # Create histograms
    min_val = min(regime_0.min(), regime_1.min())
    max_val = max(regime_0.max(), regime_1.max())
    bins = np.linspace(min_val, max_val, 100)
    
    hist_0, _ = np.histogram(regime_0, bins=bins, density=True)
    hist_1, _ = np.histogram(regime_1, bins=bins, density=True)
    
    # Overlap is the minimum at each bin
    overlap = np.sum(np.minimum(hist_0, hist_1)) / np.sum(hist_0)
    
    return overlap


def evaluate_prediction_confidence(predictions: pd.DataFrame) -> Dict:
    """Evaluate how confident the model is in its predictions.
    
    High confidence = probabilities close to 0 or 1
    Low confidence = probabilities close to 0.5 (uncertain)
    """
    # Get max probability for each prediction
    max_probs = np.maximum(predictions['state_0_prob'], predictions['state_1_prob'])
    
    results = {
        'avg_confidence': max_probs.mean(),
        'median_confidence': max_probs.median(),
        'min_confidence': max_probs.min(),
        'uncertain_predictions_pct': (max_probs < 0.7).sum() / len(max_probs) * 100,
        'very_confident_pct': (max_probs > 0.95).sum() / len(max_probs) * 100,
    }
    
    return results


def evaluate_forward_prediction(predictions: pd.DataFrame, features: pd.DataFrame, 
                                horizon: int = 5) -> Dict:
    """Evaluate if current regime predicts future volatility.
    
    Good regime detection = knowing today's regime helps predict near-term volatility.
    """
    data = pd.merge(predictions, features, left_index=True, right_index=True, how='inner')
    
    if 'rolling_std' not in data.columns:
        return {}
    
    # Compute future volatility
    data['future_vol'] = data['rolling_std'].shift(-horizon)
    data = data.dropna()
    
    # Compare prediction accuracy
    regime_0_future = data[data['predicted_state'] == 0]['future_vol']
    regime_1_future = data[data['predicted_state'] == 1]['future_vol']
    
    results = {
        f'regime_0_future_vol_{horizon}d': regime_0_future.mean(),
        f'regime_1_future_vol_{horizon}d': regime_1_future.mean(),
        f'prediction_accuracy': abs(regime_1_future.mean() - regime_0_future.mean()),
        f'prediction_correlation': data['predicted_state'].corr(data['future_vol'])
    }
    
    return results


def evaluate_regime_timing(predictions: pd.DataFrame, features: pd.DataFrame) -> Dict:
    """Evaluate if regime changes align with actual volatility changes."""
    data = pd.merge(predictions, features, left_index=True, right_index=True, how='inner')
    
    if 'rolling_std' not in data.columns:
        return {}
    
    # Find regime transitions
    regime_changes = (data['predicted_state'].diff() != 0).astype(int)
    
    # Compute volatility changes
    vol_changes = data['rolling_std'].diff().abs()
    
    # Correlation: Do regime changes happen when volatility changes?
    correlation = regime_changes.corr(vol_changes)
    
    # Average volatility change at regime transitions vs non-transitions
    vol_change_at_transition = vol_changes[regime_changes == 1].mean()
    vol_change_no_transition = vol_changes[regime_changes == 0].mean()
    
    results = {
        'regime_vol_change_correlation': correlation,
        'avg_vol_change_at_transition': vol_change_at_transition,
        'avg_vol_change_normal': vol_change_no_transition,
        'transition_sensitivity': vol_change_at_transition / vol_change_no_transition if vol_change_no_transition > 0 else 0
    }
    
    return results


def evaluate_economic_value(predictions: pd.DataFrame, features: pd.DataFrame) -> Dict:
    """Evaluate the economic value of regime predictions for trading.
    
    Compares returns from a regime-aware strategy vs buy-and-hold.
    """
    data = pd.merge(predictions, features, left_index=True, right_index=True, how='inner')
    
    if 'returns' not in data.columns:
        return {}
    
    # Strategy 1: Reduce exposure during high volatility regime
    # Assume: 100% equity in regime 0, 50% equity in regime 1
    data['strategy_weight'] = data['predicted_state'].apply(lambda x: 0.5 if x == 1 else 1.0)
    data['strategy_returns'] = data['returns'] * data['strategy_weight']
    
    # Compute cumulative returns
    buy_hold_return = (1 + data['returns']).prod() - 1
    strategy_return = (1 + data['strategy_returns']).prod() - 1
    
    # Compute Sharpe ratios (annualized, assuming 252 trading days)
    buy_hold_sharpe = (data['returns'].mean() / data['returns'].std()) * np.sqrt(252)
    strategy_sharpe = (data['strategy_returns'].mean() / data['strategy_returns'].std()) * np.sqrt(252)
    
    # Compute max drawdowns
    buy_hold_cumulative = (1 + data['returns']).cumprod()
    strategy_cumulative = (1 + data['strategy_returns']).cumprod()
    
    buy_hold_dd = (buy_hold_cumulative / buy_hold_cumulative.cummax() - 1).min()
    strategy_dd = (strategy_cumulative / strategy_cumulative.cummax() - 1).min()
    
    results = {
        'buy_hold_return': buy_hold_return * 100,
        'strategy_return': strategy_return * 100,
        'outperformance': (strategy_return - buy_hold_return) * 100,
        'buy_hold_sharpe': buy_hold_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'sharpe_improvement': strategy_sharpe - buy_hold_sharpe,
        'buy_hold_max_drawdown': buy_hold_dd * 100,
        'strategy_max_drawdown': strategy_dd * 100,
        'drawdown_improvement': (strategy_dd - buy_hold_dd) * 100
    }
    
    return results


def create_evaluation_report(artifacts_dir: str, output_dir: str) -> None:
    """Create comprehensive evaluation report."""
    logger = logging.getLogger(__name__)
    ensure_dir(output_dir)
    
    # Load data
    logger.info("Loading predictions and features...")
    predictions, features = load_predictions_and_features(artifacts_dir)
    
    # Run evaluations
    logger.info("Evaluating regime separation...")
    separation = evaluate_regime_separation(predictions, features)
    
    logger.info("Evaluating prediction confidence...")
    confidence = evaluate_prediction_confidence(predictions)
    
    logger.info("Evaluating forward prediction...")
    forward_5d = evaluate_forward_prediction(predictions, features, horizon=5)
    forward_20d = evaluate_forward_prediction(predictions, features, horizon=20)
    
    logger.info("Evaluating regime timing...")
    timing = evaluate_regime_timing(predictions, features)
    
    logger.info("Evaluating economic value...")
    economic = evaluate_economic_value(predictions, features)
    
    # Print report
    print("\n" + "="*80)
    print("REGIME DETECTION MODEL EVALUATION REPORT")
    print("="*80)
    
    print("\n1. REGIME SEPARATION QUALITY")
    print("-" * 80)
    if separation:
        print(f"   Regime 0 (Low Vol) avg:    {separation['regime_0_mean_vol']:.4f}")
        print(f"   Regime 1 (High Vol) avg:   {separation['regime_1_mean_vol']:.4f}")
        print(f"   Difference:                +{separation['relative_difference_pct']:.1f}%")
        print(f"   Cohen's d (effect size):   {separation['cohens_d']:.2f} ", end="")
        if separation['cohens_d'] > 0.8:
            print("(LARGE - Excellent)")
        elif separation['cohens_d'] > 0.5:
            print("(MEDIUM - Good)")
        else:
            print("(SMALL - Weak)")
        print(f"   P-value:                   {separation['p_value']:.2e} ", end="")
        print("(Significant)" if separation['p_value'] < 0.01 else "(Not significant)")
        print(f"   Overlap coefficient:       {separation['overlap_coefficient']:.2%} ", end="")
        print("(Lower is better)")
    
    print("\n2. PREDICTION CONFIDENCE")
    print("-" * 80)
    print(f"   Average confidence:        {confidence['avg_confidence']:.1%}")
    print(f"   Very confident (>95%):     {confidence['very_confident_pct']:.1f}% of predictions")
    print(f"   Uncertain (<70%):          {confidence['uncertain_predictions_pct']:.1f}% of predictions")
    print(f"   → Model is ", end="")
    if confidence['avg_confidence'] > 0.9:
        print("VERY CONFIDENT in predictions")
    elif confidence['avg_confidence'] > 0.75:
        print("MODERATELY CONFIDENT in predictions")
    else:
        print("UNCERTAIN in predictions")
    
    print("\n3. FORWARD PREDICTION ABILITY")
    print("-" * 80)
    if forward_5d:
        print(f"   5-day ahead:")
        print(f"     Regime 0 → Future vol:   {forward_5d['regime_0_future_vol_5d']:.4f}")
        print(f"     Regime 1 → Future vol:   {forward_5d['regime_1_future_vol_5d']:.4f}")
        print(f"     Predictive power:        {forward_5d['prediction_accuracy']:.4f}")
        print(f"     Correlation:             {forward_5d['prediction_correlation']:.3f}")
    if forward_20d:
        print(f"   20-day ahead:")
        print(f"     Regime 0 → Future vol:   {forward_20d['regime_0_future_vol_20d']:.4f}")
        print(f"     Regime 1 → Future vol:   {forward_20d['regime_1_future_vol_20d']:.4f}")
        print(f"     Predictive power:        {forward_20d['prediction_accuracy']:.4f}")
    
    print("\n4. REGIME TIMING QUALITY")
    print("-" * 80)
    if timing:
        print(f"   Correlation with vol changes:  {timing['regime_vol_change_correlation']:.3f}")
        print(f"   Vol change at transitions:     {timing['avg_vol_change_at_transition']:.4f}")
        print(f"   Vol change normally:           {timing['avg_vol_change_normal']:.4f}")
        print(f"   Transition sensitivity:        {timing['transition_sensitivity']:.2f}x")
        print(f"   → Regime changes are ", end="")
        if timing['transition_sensitivity'] > 2:
            print("WELL-TIMED with volatility shifts")
        elif timing['transition_sensitivity'] > 1.2:
            print("SOMEWHAT ALIGNED with volatility shifts")
        else:
            print("POORLY TIMED (might be too noisy)")
    
    print("\n5. ECONOMIC VALUE (Trading Performance)")
    print("-" * 80)
    if economic:
        print(f"   Buy & Hold return:         {economic['buy_hold_return']:+.2f}%")
        print(f"   Regime-aware strategy:     {economic['strategy_return']:+.2f}%")
        print(f"   Outperformance:            {economic['outperformance']:+.2f}%")
        print()
        print(f"   Buy & Hold Sharpe:         {economic['buy_hold_sharpe']:.2f}")
        print(f"   Strategy Sharpe:           {economic['strategy_sharpe']:.2f}")
        print(f"   Sharpe improvement:        {economic['sharpe_improvement']:+.2f}")
        print()
        print(f"   Buy & Hold max drawdown:   {economic['buy_hold_max_drawdown']:.2f}%")
        print(f"   Strategy max drawdown:     {economic['strategy_max_drawdown']:.2f}%")
        print(f"   Drawdown reduction:        {economic['drawdown_improvement']:+.2f}%")
        print()
        print(f"   → Strategy is ", end="")
        if economic['outperformance'] > 5 and economic['sharpe_improvement'] > 0.2:
            print("HIGHLY VALUABLE for trading")
        elif economic['outperformance'] > 0 or economic['sharpe_improvement'] > 0:
            print("SOMEWHAT VALUABLE for trading")
        else:
            print("NOT ADDING VALUE (regime info not useful for trading)")
    
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    # Scoring
    score = 0
    max_score = 5
    
    if separation and separation['cohens_d'] > 0.5:
        score += 1
    if confidence['avg_confidence'] > 0.8:
        score += 1
    if forward_5d and forward_5d['prediction_accuracy'] > 0.01:
        score += 1
    if timing and timing['transition_sensitivity'] > 1.5:
        score += 1
    if economic and economic['sharpe_improvement'] > 0:
        score += 1
    
    print(f"Model Quality Score: {score}/{max_score}")
    print()
    
    if score >= 4:
        print("✅ EXCELLENT - Model predictions are reliable and economically valuable")
    elif score >= 3:
        print("✓ GOOD - Model predictions are useful but could be improved")
    elif score >= 2:
        print("⚠ FAIR - Model captures some signal but needs refinement")
    else:
        print("❌ POOR - Model predictions may not be reliable for trading")
    
    print("="*80)
    
    # Save detailed results
    from regime_lab.utils.config import save_json
    all_results = {
        'separation': separation,
        'confidence': confidence,
        'forward_5d': forward_5d,
        'forward_20d': forward_20d,
        'timing': timing,
        'economic': economic,
        'overall_score': f"{score}/{max_score}"
    }
    
    output_file = Path(output_dir) / "evaluation_results.json"
    save_json(all_results, str(output_file))
    logger.info(f"Saved detailed results to {output_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate regime detection predictions")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory containing model artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting model evaluation...")
        create_evaluation_report(args.artifacts_dir, args.output_dir)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

