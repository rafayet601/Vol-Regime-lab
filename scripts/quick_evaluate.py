#!/usr/bin/env python3
"""Quick evaluation of regime detection predictions."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Quick evaluate regime predictions")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory containing model artifacts"
    )
    
    args = parser.parse_args()
    
    # Load data
    pred = pd.read_csv(f"{args.artifacts_dir}/posteriors.csv")
    feat = pd.read_csv(f"{args.artifacts_dir}/features.csv")
    
    # Merge on date
    data = pd.merge(pred, feat, on='date', how='inner')
    
    print("\n" + "="*80)
    print("REGIME DETECTION MODEL EVALUATION")
    print("="*80)
    
    # 1. REGIME SEPARATION QUALITY
    print("\n1. REGIME SEPARATION (Do the regimes have different volatility?)")
    print("-" * 80)
    
    regime_0_vol = data[data['predicted_state'] == 0]['rolling_std']
    regime_1_vol = data[data['predicted_state'] == 1]['rolling_std']
    
    mean_0 = regime_0_vol.mean()
    mean_1 = regime_1_vol.mean()
    std_0 = regime_0_vol.std()
    std_1 = regime_1_vol.std()
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
    cohens_d = (mean_1 - mean_0) / pooled_std
    
    # T-test
    t_stat, p_value = stats.ttest_ind(regime_0_vol, regime_1_vol)
    
    print(f"   Regime 0 (Low Vol):        {mean_0:.4f} ± {std_0:.4f}")
    print(f"   Regime 1 (High Vol):       {mean_1:.4f} ± {std_1:.4f}")
    print(f"   Difference:                {mean_1 - mean_0:.4f} ({((mean_1-mean_0)/mean_0*100):+.1f}%)")
    print(f"   Cohen's d:                 {cohens_d:.2f} ", end="")
    if cohens_d > 0.8:
        print("✅ LARGE (Excellent separation)")
    elif cohens_d > 0.5:
        print("✓ MEDIUM (Good separation)")
    elif cohens_d > 0.2:
        print("⚠ SMALL (Weak separation)")
    else:
        print("❌ NEGLIGIBLE (Poor separation)")
    print(f"   P-value:                   {p_value:.2e} {'✅ Significant' if p_value < 0.01 else '❌ Not significant'}")
    
    # 2. PREDICTION CONFIDENCE
    print("\n2. PREDICTION CONFIDENCE (How certain is the model?)")
    print("-" * 80)
    
    max_probs = np.maximum(data['state_0_prob'], data['state_1_prob'])
    avg_confidence = max_probs.mean()
    very_confident = (max_probs > 0.95).sum() / len(max_probs) * 100
    uncertain = (max_probs < 0.7).sum() / len(max_probs) * 100
    
    print(f"   Average confidence:        {avg_confidence:.1%}")
    print(f"   Very confident (>95%):     {very_confident:.1f}%")
    print(f"   Uncertain (<70%):          {uncertain:.1f}%")
    if avg_confidence > 0.9:
        print(f"   Assessment:                ✅ Model is VERY CONFIDENT")
    elif avg_confidence > 0.75:
        print(f"   Assessment:                ✓ Model is MODERATELY CONFIDENT")
    else:
        print(f"   Assessment:                ⚠ Model is UNCERTAIN")
    
    # 3. REGIME CHARACTERISTICS
    print("\n3. REGIME CHARACTERISTICS")
    print("-" * 80)
    
    # Get returns by regime (use abs_returns as proxy)
    regime_0_abs_ret = data[data['predicted_state'] == 0]['abs_returns']
    regime_1_abs_ret = data[data['predicted_state'] == 1]['abs_returns']
    
    print(f"   Regime 0 (Low Vol):")
    print(f"     Days:                    {len(regime_0_vol)} ({len(regime_0_vol)/len(data)*100:.1f}%)")
    print(f"     Avg volatility:          {mean_0:.4f}")
    print(f"     Avg abs return:          {regime_0_abs_ret.mean():.4f}")
    print(f"   Regime 1 (High Vol):")
    print(f"     Days:                    {len(regime_1_vol)} ({len(regime_1_vol)/len(data)*100:.1f}%)")
    print(f"     Avg volatility:          {mean_1:.4f}")
    print(f"     Avg abs return:          {regime_1_abs_ret.mean():.4f}")
    
    # 4. REGIME PERSISTENCE
    print("\n4. REGIME PERSISTENCE (How stable are the regimes?)")
    print("-" * 80)
    
    from itertools import groupby
    
    states = data['predicted_state'].values
    runs = [(state, len(list(group))) for state, group in groupby(states)]
    run_lengths = [length for _, length in runs]
    transitions = len(runs) - 1
    
    print(f"   Total regime switches:     {transitions}")
    print(f"   Average duration:          {np.mean(run_lengths):.1f} days")
    print(f"   Median duration:           {np.median(run_lengths):.0f} days")
    print(f"   Min duration:              {np.min(run_lengths)} days")
    print(f"   Max duration:              {np.max(run_lengths)} days")
    
    if np.mean(run_lengths) > 20:
        print(f"   Assessment:                ✅ STABLE regimes (good for swing trading)")
    elif np.mean(run_lengths) > 5:
        print(f"   Assessment:                ✓ MODERATE stability (good for weekly strategies)")
    else:
        print(f"   Assessment:                ⚠ UNSTABLE regimes (only for daily trading)")
    
    # 5. FORWARD PREDICTION
    print("\n5. FORWARD PREDICTION (Does current regime predict future volatility?)")
    print("-" * 80)
    
    # Compute future volatility (5 days ahead)
    data['future_vol_5d'] = data['rolling_std'].shift(-5)
    data_clean = data.dropna()
    
    future_0 = data_clean[data_clean['predicted_state'] == 0]['future_vol_5d'].mean()
    future_1 = data_clean[data_clean['predicted_state'] == 1]['future_vol_5d'].mean()
    
    print(f"   If in Regime 0 now:")
    print(f"     Expected vol in 5 days:  {future_0:.4f}")
    print(f"   If in Regime 1 now:")
    print(f"     Expected vol in 5 days:  {future_1:.4f}")
    print(f"   Predictive difference:     {future_1 - future_0:.4f} ({((future_1-future_0)/future_0*100):+.1f}%)")
    
    if abs((future_1-future_0)/future_0) > 0.2:
        print(f"   Assessment:                ✅ STRONG predictive power")
    elif abs((future_1-future_0)/future_0) > 0.1:
        print(f"   Assessment:                ✓ MODERATE predictive power")
    else:
        print(f"   Assessment:                ⚠ WEAK predictive power")
    
    # 6. OVERALL SCORE
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    score = 0
    if cohens_d > 0.5:
        score += 1
    if p_value < 0.01:
        score += 1
    if avg_confidence > 0.8:
        score += 1
    if np.mean(run_lengths) > 3:
        score += 1
    if abs((future_1-future_0)/future_0) > 0.1:
        score += 1
    
    print(f"\n   Model Quality Score: {score}/5\n")
    
    if score >= 4:
        print("   ✅ EXCELLENT - Model is reliable and actionable")
        print("      → Regime predictions are trustworthy")
        print("      → Can be used for trading decisions")
    elif score >= 3:
        print("   ✓ GOOD - Model is useful with some caveats")
        print("      → Regime predictions have value")
        print("      → Use with additional confirmation")
    elif score >= 2:
        print("   ⚠ FAIR - Model captures some signal")
        print("      → Regime predictions are noisy")
        print("      → Consider refinement before trading")
    else:
        print("   ❌ POOR - Model needs significant improvement")
        print("      → Regime predictions may not be reliable")
        print("      → Do not use for trading without major changes")
    
    print("\n" + "="*80)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 80)
    print(f"✓ Regime 1 has {((mean_1-mean_0)/mean_0*100):.0f}% higher volatility than Regime 0")
    print(f"✓ Model is {avg_confidence:.0%} confident in its predictions")
    print(f"✓ Regimes last an average of {np.mean(run_lengths):.1f} days")
    print(f"✓ Current regime predicts {abs((future_1-future_0)/future_0*100):.0f}% difference in future volatility")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 80)
    if np.mean(run_lengths) < 5:
        print("⚠ Consider smoothing regimes to reduce daily switching")
        print(f"  → Run: python scripts/smooth_regimes.py --min-duration 10")
    if cohens_d < 0.5:
        print("⚠ Increase rolling window size for clearer regime separation")
    if score >= 3:
        print("✓ Model is ready for backtesting with trading strategies")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

