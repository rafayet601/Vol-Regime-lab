#!/usr/bin/env python3
"""
Regime Lab Demo Script (v2.0 API)
==================================
This script demonstrates all capabilities of the Regime Lab framework
using the v2.0 functional feature API.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("🎯 REGIME LAB DEMO - S&P 500 Regime Detection (v2.0)")
print("="*70)
print()

# =============================================================================
# DEMO 1: Data Loading and Feature Engineering
# =============================================================================
print("📊 DEMO 1: Data Loading and Feature Engineering")
print("-" * 70)

from regime_lab.data.features import (
    build_features, get_feature_cols, load_spx_data, load_vix_data
)

# Load S&P 500 data (using smaller date range for demo)
print("\n📥 Loading S&P 500 data (2020-2024)...")
try:
    prices, returns = load_spx_data(
        start_date="2020-01-01",
        end_date="2024-01-01",
    )
    print(f"✓ Loaded {len(prices)} price observations")
    print(f"✓ Computed {len(returns)} return observations")
    print(f"\nPrice data columns: {list(prices.columns)}")
    print(f"Price range: ${prices['close'].min():.2f} - ${prices['close'].max():.2f}")
except Exception as e:
    print(f"⚠️  Data download in progress, using synthetic data for demo: {e}")
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='B')
    np.random.seed(42)
    ret_vals = np.random.normal(0.0005, 0.012, len(dates))
    returns = pd.Series(ret_vals, index=dates, name="returns")
    prices = pd.DataFrame({
        'open': 3000 * np.cumprod(1 + ret_vals),
        'high': 3000 * np.cumprod(1 + ret_vals) * 1.005,
        'low':  3000 * np.cumprod(1 + ret_vals) * 0.995,
        'close': 3000 * np.cumprod(1 + ret_vals),
    }, index=dates)
    print(f"✓ Created {len(returns)} synthetic observations")

# Load VIX data for Tier 2 features
print("\n📥 Loading VIX data...")
try:
    vix_df = load_vix_data(start_date="2020-01-01", end_date="2024-01-01")
    print(f"✓ Loaded VIX data: {len(vix_df)} days")
except Exception as e:
    print(f"⚠️  VIX data unavailable, using Tier 1 features only: {e}")
    vix_df = None

# Engineer features using v2.0 functional API
print("\n🔧 Building feature matrix (v2.0 API)...")
features = build_features(returns, prices=prices, vix_df=vix_df)
feature_cols = get_feature_cols(features, tier=2 if vix_df is not None else 1)

print(f"✓ Built {len(features.columns)} columns")
print(f"✓ Selected {len(feature_cols)} Tier {'2' if vix_df is not None else '1'} features: {feature_cols}")
print(f"✓ Valid observations: {len(features)} (after removing NaN)")

# Display feature statistics
print("\n📈 Feature Statistics:")
print(features[feature_cols].describe().to_string())

print("\n" + "="*70)
input("Press Enter to continue to Demo 2...")
print()

# =============================================================================
# DEMO 2: HMM Model Training
# =============================================================================
print("🎯 DEMO 2: HMM Model Training with Baum-Welch")
print("-" * 70)

from regime_lab.models.hmm_studentt import StudentTHMM

# Prepare features for model
X = features[feature_cols].values

print(f"📊 Training data shape: {X.shape}")
print(f"   - {X.shape[0]} observations")
print(f"   - {X.shape[1]} features: {feature_cols}")

# Initialize and train HMM (v2.0 API: no n_features or df args)
print("\n🤖 Initializing 2-state Student-t HMM...")
model = StudentTHMM(
    n_states=2,
    fix_nu=False,
    max_iter=50,
    tol=1e-6,
    random_seed=42,
)

print("✓ Model initialized with Student-t emissions (ν learned via Newton-Raphson)")

print("\n🔄 Training model with Baum-Welch algorithm...")
print("   (This may take a minute...)")

try:
    model.fit(X)
    print("✓ Model training completed successfully!")

    # Display model parameters
    print("\n📊 Model Parameters:")
    model_summary = model.get_model_summary()
    print(f"   States: {model_summary['n_states']}")
    print(f"   Converged: {model_summary['converged']} in {model_summary['n_iter']} iterations")
    print(f"   Log-likelihood: {model_summary['log_likelihood']:.2f}")
    print(f"   BIC: {model_summary['bic']:.2f}")

    print("\n🔀 Transition Matrix:")
    A = np.array(model_summary['A'])
    print(f"   State 0 → State 0: {A[0,0]:.3f}")
    print(f"   State 0 → State 1: {A[0,1]:.3f}")
    print(f"   State 1 → State 0: {A[1,0]:.3f}")
    print(f"   State 1 → State 1: {A[1,1]:.3f}")

    print("\n🎨 State Characteristics:")
    mu = np.array(model_summary['mu'])
    sigma = np.array(model_summary['sigma'])
    nu = np.array(model_summary['nu'])
    for i in range(2):
        regime_type = "Low Volatility" if i == 0 else "High Volatility"
        print(f"\n   State {i} ({regime_type}):")
        for j, col in enumerate(feature_cols):
            print(f"     {col}: μ={mu[i,j]:.4f}, σ={sigma[i,j]:.4f}")
        print(f"     ν (degrees of freedom): {nu[i]:.2f}")

except Exception as e:
    print(f"⚠️  Model training encountered an issue: {e}")
    print("   Continuing with demo using simulated predictions...")

print("\n" + "="*70)
input("Press Enter to continue to Demo 3...")
print()

# =============================================================================
# DEMO 3: State Prediction and Analysis
# =============================================================================
print("🔮 DEMO 3: State Prediction and Regime Analysis")
print("-" * 70)

print("🎯 Predicting hidden states using Viterbi algorithm...")
try:
    predicted_states = model.predict(X)
    state_probabilities = model.predict_proba(X)
    print(f"✓ Generated predictions for {len(predicted_states)} observations")
except Exception as e:
    print(f"⚠️  Using simulated predictions: {e}")
    predicted_states = (np.random.rand(len(X)) > 0.7).astype(int)
    state_probabilities = np.column_stack([
        1 - predicted_states + np.random.rand(len(X)) * 0.1,
        predicted_states + np.random.rand(len(X)) * 0.1
    ])

# Create results DataFrame
results_df = pd.DataFrame({
    'date': features.index,
    'predicted_state': predicted_states,
    'state_0_prob': state_probabilities[:, 0],
    'state_1_prob': state_probabilities[:, 1],
    'returns': features['returns'].values,
})

print("\n📊 Regime Distribution:")
regime_counts = results_df['predicted_state'].value_counts().sort_index()
for state, count in regime_counts.items():
    pct = (count / len(results_df)) * 100
    regime_name = "Low Volatility" if state == 0 else "High Volatility"
    print(f"   State {state} ({regime_name}): {count} obs ({pct:.1f}%)")

# Transition analysis
print("\n🔄 Regime Transitions:")
transitions = (results_df['predicted_state'].diff() != 0).sum()
avg_duration = len(results_df) / (transitions + 1)
print(f"   Total transitions: {transitions}")
print(f"   Average regime duration: {avg_duration:.1f} days")

# Performance by regime
print("\n💰 Returns by Regime:")
for state in [0, 1]:
    regime_returns = results_df[results_df['predicted_state'] == state]['returns']
    regime_name = "Low Volatility" if state == 0 else "High Volatility"
    if len(regime_returns) > 0 and regime_returns.std() > 0:
        print(f"\n   State {state} ({regime_name}):")
        print(f"     Mean return: {regime_returns.mean()*100:.4f}%")
        print(f"     Std dev: {regime_returns.std()*100:.4f}%")
        print(f"     Sharpe ratio: {(regime_returns.mean() / regime_returns.std()):.3f}")

print("\n" + "="*70)
input("Press Enter to continue to Demo 4...")
print()

# =============================================================================
# DEMO 4: Diagnostics and Statistics
# =============================================================================
print("📋 DEMO 4: Comprehensive Diagnostics")
print("-" * 70)

from regime_lab.eval.diagnostics import RegimeDiagnostics

diagnostics = RegimeDiagnostics(regime_column="predicted_state")

print("🔍 Computing comprehensive diagnostics...")
diagnostic_report = diagnostics.generate_diagnostic_report(
    results_df,
    feature_data=features
)

print("\n" + "="*70)
print("📊 DIAGNOSTIC REPORT")
print("="*70)

diagnostics.print_diagnostic_summary(diagnostic_report)

# Additional insights
print("\n🎯 Regime Characteristics by Features:")
regime_chars = diagnostic_report['regime_characteristics']
for regime_key, chars in regime_chars.items():
    regime_num = int(regime_key.split('_')[1])
    regime_name = "Low Volatility" if regime_num == 0 else "High Volatility"
    print(f"\n{regime_name} Regime:")
    print(f"  Frequency: {chars['frequency']} observations ({chars['percentage']:.1f}%)")
    if 'feature_statistics' in chars:
        print("  Feature averages:")
        for feature in feature_cols[:3]:
            if feature in chars['feature_statistics']:
                stats = chars['feature_statistics'][feature]
                print(f"    {feature}: {stats['mean']:.6f} (±{stats['std']:.6f})")

print("\n" + "="*70)
input("Press Enter to continue to Demo 5...")
print()

# =============================================================================
# DEMO 5: Visualization Capabilities
# =============================================================================
print("📈 DEMO 5: Visualization Capabilities")
print("-" * 70)

from regime_lab.plotting.regimes import RegimePlotter
from regime_lab.utils.config import ensure_dir

# Create output directory
ensure_dir("./demo_output")

plotter = RegimePlotter(figsize=(15, 10))

print("🎨 Creating visualizations...")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # 1. Regime transitions plot
    print("\n1️⃣  Creating regime transitions plot...")
    fig1 = plotter.plot_regime_transitions(
        results_df,
        title="S&P 500 Regime Transitions (2020-2024)",
        save_path="./demo_output/regime_transitions.png"
    )
    print("   ✓ Saved: demo_output/regime_transitions.png")

    # 2. Feature analysis plot
    print("\n2️⃣  Creating feature analysis plot...")
    fig2 = plotter.plot_feature_analysis(
        features,
        results_df,
        feature_columns=feature_cols[:3],
        title="Feature Distributions by Regime",
        save_path="./demo_output/feature_analysis.png"
    )
    print("   ✓ Saved: demo_output/feature_analysis.png")

    # 3. Volatility comparison
    print("\n3️⃣  Creating volatility comparison plot...")
    combined_data = features.copy()
    combined_data['predicted_state'] = results_df['predicted_state'].values
    fig3 = plotter.plot_volatility_comparison(
        combined_data,
        title="Volatility Measures Over Time",
        save_path="./demo_output/volatility_comparison.png"
    )
    print("   ✓ Saved: demo_output/volatility_comparison.png")

    print("\n✅ All visualizations created successfully!")
    print("   Check the 'demo_output' folder for the generated plots.")

except Exception as e:
    print(f"⚠️  Visualization error (expected in some environments): {e}")
    print("   You can generate plots using the plot_regimes.py script later.")

print("\n" + "="*70)
print()

# =============================================================================
# DEMO 6: Saving and Loading Models
# =============================================================================
print("💾 DEMO 6: Model Persistence")
print("-" * 70)

from regime_lab.utils.io import save_dataframe, save_pickle
from regime_lab.utils.config import save_json

print("💾 Saving model and results...")

try:
    # Save results
    save_dataframe(results_df, "./demo_output/predictions.csv")
    print("✓ Predictions saved to: demo_output/predictions.csv")

    # Save diagnostics
    save_json(diagnostic_report, "./demo_output/diagnostics.json")
    print("✓ Diagnostics saved to: demo_output/diagnostics.json")

    # Save feature data
    save_dataframe(features, "./demo_output/features.csv")
    print("✓ Features saved to: demo_output/features.csv")

    print("\n✅ All artifacts saved successfully!")

except Exception as e:
    print(f"⚠️  Save error: {e}")

print("\n" + "="*70)
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("="*70)
print("🎉 DEMO COMPLETE - Summary of Capabilities")
print("="*70)
print()
print("✅ Demonstrated Features:")
print()
print("1️⃣  📊 Data Loading")
print("   - S&P 500 data retrieval via yfinance")
print("   - VIX / VIX9D / VIX3M term structure data")
print("   - Log returns computation")
print()
print("2️⃣  🔧 Feature Engineering (v2.0)")
print("   - Multi-horizon rolling vol (5-/20-/60-day)")
print("   - VRP (Bollerslev, Tauchen & Zhou 2009)")
print("   - VIX term structure: spot_ratio, term_ratio")
print("   - Downside vol & Parkinson estimator")
print("   - Tiered feature selection")
print()
print("3️⃣  🤖 Student-t HMM")
print("   - K-state HMM with diagonal Student-t emissions")
print("   - Baum-Welch EM with log-space forward-backward")
print("   - Newton-Raphson M-step for ν")
print("   - Canonical label-switching resolution")
print()
print("4️⃣  🔮 Predictions")
print("   - Viterbi MAP state sequence")
print("   - Forward-backward state probabilities")
print("   - AIC/BIC model selection")
print()
print("5️⃣  📋 Diagnostics")
print("   - Duration statistics")
print("   - Persistence tests")
print("   - Transition analysis")
print("   - Feature characteristics by regime")
print()
print("6️⃣  📈 Visualizations")
print("   - Regime transition plots")
print("   - Feature distribution analysis")
print("   - Volatility comparisons")
print("   - Transition matrices")
print()
print("7️⃣  💾 Persistence")
print("   - CSV / JSON artifact export")
print("   - Full reproducibility")
print()
print("="*70)
print("📁 Output Files (in demo_output/):")
print("   - predictions.csv           (State predictions)")
print("   - diagnostics.json          (Comprehensive diagnostics)")
print("   - features.csv              (Engineered features)")
print("   - regime_transitions.png    (Transition plot)")
print("   - feature_analysis.png      (Feature distributions)")
print("   - volatility_comparison.png (Volatility plot)")
print()
print("="*70)
print("🚀 Next Steps:")
print()
print("   1. Train on full dataset:")
print("      $ python scripts/train_hmm.py")
print()
print("   2. Generate comprehensive plots:")
print("      $ python scripts/plot_regimes.py")
print()
print("   3. Run tests:")
print("      $ pytest tests/ -v")
print()
print("="*70)
print("✨ Thank you for using Regime Lab v2.0!")
print("="*70)
