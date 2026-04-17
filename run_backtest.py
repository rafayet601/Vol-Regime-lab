#!/usr/bin/env python3
"""
Walk-Forward Backtesting Script
================================
Runs a walk-forward backtest of the S&P 500 Student-t HMM across various
regime-aware trading strategies and benchmarks them.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from regime_lab.backtest.walk_forward import WalkForwardBacktester, print_backtest_summary
from regime_lab.backtest.strategy import build_strategy_suite
from regime_lab.data.features import build_features, get_feature_cols, load_spx_data, load_vix_data
from regime_lab.models.hmm_studentt import StudentTHMM

def get_model_factory():
    """Returns a fresh model initialized without data, ready to be fit on train slices."""
    def factory():
        return StudentTHMM(
            n_states=2,
            max_iter=50,
            tol=1e-5,
            random_seed=42
        )
    return factory

def main():
    print("="*70)
    print("🚀 LAUNCHING WALK-FORWARD BACKTEST")
    print("="*70)

    # 1. Load Data
    print("\n📥 Loading Historical Data (2010 - 2024)...")
    prices, returns = load_spx_data(start_date="2010-01-01", end_date="2024-01-01")
    
    try:
        vix_df = load_vix_data(start_date="2010-01-01", end_date="2024-01-01")
        tier = 2
    except Exception as e:
        print(f"⚠️ Could not load VIX: {e}. Falling back to Tier 1.")
        vix_df = None
        tier = 1

    # 2. Build Features
    print("\n🔧 Building Feature Matrix...")
    features = build_features(returns, prices=prices, vix_df=vix_df)
    feature_cols = get_feature_cols(features, tier=tier)
    print(f"   Using {len(feature_cols)} features: {feature_cols}")

    # Ensure 'returns' is in the features dataframe for the backtester PnL calculation
    features['returns'] = returns.loc[features.index]

    # 3. Setup Strategies
    suites = build_strategy_suite(
        theta_high=0.65,
        theta_low=0.35,
        vol_target=0.10,     # Targeting 10% annualized vol
        max_leverage=2.0,    # Max 2x leverage
        hold_period=5
    )

    print("\n📈 Initializing Strategies to Backtest:")
    for name, _ in suites.items():
        print(f"   - {name}")

    # 4. Run Walk-Forward Backtester
    print("\n⏳ Running Walk-Forward Process (this may take a couple of minutes)...")
    print("   Settings: 3 years train window (756 days), 1 month step (21 days)\n")
    
    for name, strategy_func in suites.items():
        print(f"--- Testing Strategy: {name.upper()} ---")
        
        backtester = WalkForwardBacktester(
            model_factory=get_model_factory(),
            strategy=strategy_func,
            train_window=756,      # Roughly 3 years
            step=21,               # Roughly 1 month
            grow_window=False,     # Rolling window instead of expanding
            feature_cols=feature_cols,
            normalize=True
        )

        try:
            result = backtester.run(features=features, returns=features['returns'])
            print_backtest_summary(result)
        except Exception as e:
            print(f"   ❌ Backtest failed for {name}: {e}\n")

if __name__ == "__main__":
    main()
