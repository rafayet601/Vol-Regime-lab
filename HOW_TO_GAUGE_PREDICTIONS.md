# How to Gauge Regime Detection Predictions

This guide explains how to evaluate whether your regime detection model's predictions are useful and reliable.

## Quick Start

Run the evaluation script:

```bash
python scripts/quick_evaluate.py --artifacts-dir artifacts
```

This will give you a **Model Quality Score** from 0/5 to 5/5 and specific recommendations.

---

## What the Evaluation Checks

### 1. **Regime Separation Quality** ⭐ MOST IMPORTANT

**Question:** Do the two regimes have meaningfully different volatility?

**Metrics:**
- **Cohen's d (Effect Size)**: Measures how different the regimes are
  - `> 0.8` = Large effect (EXCELLENT)
  - `0.5-0.8` = Medium effect (GOOD)
  - `0.2-0.5` = Small effect (WEAK)
  - `< 0.2` = Negligible (POOR)
  
- **P-value**: Statistical significance
  - `< 0.01` = Significant difference ✅
  - `> 0.01` = Not statistically different ❌

**Latest Results (After Improvements):**
```
ORIGINAL MODEL (rolling_window=20):
  Cohen's d: -0.06 (NEGLIGIBLE)
  P-value: 0.0327 (NOT SIGNIFICANT)
  Score: 1/5

SMOOTHED (10-day min):
  Cohen's d: -0.10 (NEGLIGIBLE)
  P-value: 0.000151 (SIGNIFICANT)
  Score: 2/5

RETRAINED (rolling_window=50):
  Cohen's d: -0.05 (NEGLIGIBLE)
  P-value: 0.088 (NOT SIGNIFICANT)
  Score: 1/5

RETRAINED + SMOOTHED:
  Cohen's d: -0.09 (NEGLIGIBLE)
  P-value: 0.000474 (SIGNIFICANT)
  Score: 2/5
```
❌ **Problem:** Even with improvements, regimes are barely different! Regime 0 has HIGHER volatility than Regime 1 (labels are backwards).

**Why this matters:** If regimes don't differ in volatility, they're useless for trading.

---

### 2. **Prediction Confidence**

**Question:** How certain is the model about its predictions?

**Metrics:**
- **Average confidence**: Should be > 80%
- **Uncertain predictions**: Should be < 20%

**Your Current Result:**
```
Average confidence: 100%
Very confident: 100%
```
✅ **Good:** Model is very confident (perhaps TOO confident given poor separation)

**Why this matters:** High confidence is only useful if the predictions are actually correct!

---

### 3. **Regime Persistence**

**Question:** How long do regimes last?

**Metrics:**
- **Average duration**: 
  - `> 20 days` = Stable (good for swing trading)
  - `5-20 days` = Moderate (good for weekly strategies)
  - `< 5 days` = Unstable (only for daily trading)

**Your Current Result:**
```
Average duration: 1.9 days
Median duration: 1 day
Max duration: 9 days
Transitions: 3,050 in 23 years
```
❌ **Problem:** Regimes switch almost every day! This causes:
- Excessive trading costs
- Whipsaw (buying/selling constantly)
- Noise rather than signal

**Why this matters:** You can't realistically trade on 1-day regimes.

---

### 4. **Forward Prediction**

**Question:** Does knowing today's regime help predict future volatility?

**Metrics:**
- **Predictive difference**: How different is future volatility between regimes?
  - `> 20%` = Strong predictive power
  - `10-20%` = Moderate predictive power
  - `< 10%` = Weak predictive power

**Your Current Result:**
```
Regime 0 → Future vol in 5 days: 0.1697
Regime 1 → Future vol in 5 days: 0.1582
Difference: -6.8%
```
⚠️ **Weak:** Current regime barely predicts future volatility

**Why this matters:** If current regime doesn't predict the future, it's not useful for forward-looking decisions.

---

## Your Current Model: Overall Assessment

### Score: **2/5** ⚠️ (Best with Smoothing)

**Status:** FAIR - Model captures some signal but needs further improvement

**Current Best:** Smoothed regimes (10-day minimum)
- Available in: `artifacts_smoothed_10day/` and `artifacts_rw50_smoothed_10day/`

**Main Issues:**
1. ❌ **Labels are backwards**: "Low Vol" regime actually has higher volatility
2. ❌ **Poor separation**: Cohen's d = -0.09 to -0.10 (regimes barely differ)
3. ❌ **Too unstable**: Switches every 2.0 days on average (only 6.6% improvement from smoothing)
4. ❌ **Weak predictive power**: Only 5-7% difference in future volatility

---

## How to Improve Your Model

### Option 1: Smooth the Regimes (Quick Fix)

Force minimum 10-day regime duration to reduce noise:

```bash
python scripts/smooth_regimes.py --input-dir artifacts --output-dir artifacts_smoothed_10day --min-duration 10
```

Then evaluate the smoothed version:
```bash
python scripts/quick_evaluate.py --artifacts-dir artifacts_smoothed_10day
```

### Option 2: Retrain with Better Features

Edit `configs/hmm_spx_studentt.yaml`:
```yaml
features:
  rolling_window: 50  # Increase from 20 to 50 for smoother features
  volatility_method: "ewm"  # Use exponential weighting
```

Then retrain:
```bash
python scripts/train_hmm.py --config configs/hmm_spx_studentt.yaml --output-dir artifacts_v2
```

### Option 3: Use More States

Try 3 states instead of 2 (Low, Medium, High volatility):
```yaml
model:
  n_states: 3  # Instead of 2
```

---

## What Makes a Good Model?

### Target Metrics

| Metric | Target | Why |
|--------|--------|-----|
| Cohen's d | > 0.5 | Clear regime separation |
| P-value | < 0.01 | Statistically significant |
| Avg Confidence | > 80% | Model is sure of predictions |
| Avg Duration | 5-20 days | Tradeable but not too slow |
| Predictive Power | > 10% | Current regime predicts future |

### Example of a Good Model

```
Cohen's d:                 0.85 (LARGE)
P-value:                   < 0.001 (SIGNIFICANT)
Average confidence:        92%
Average duration:          12 days
Regime 0 → Future vol:     0.12
Regime 1 → Future vol:     0.28
Difference:                +133% (STRONG)

Score: 5/5 ✅ EXCELLENT
```

---

## Practical Use Cases by Score

### Score 4-5/5: ✅ Ready for Trading
- Use for actual trading decisions
- Adjust position sizing based on regime
- Set stop-losses based on regime volatility
- Rebalance portfolio when regimes change

### Score 3/5: ✓ Use with Caution
- Combine with other indicators
- Use for risk management, not entry/exit signals
- Backtest thoroughly before live trading

### Score 1-2/5: ❌ Not Ready
- Do NOT trade on these predictions
- Focus on improving the model first
- Use only for research/understanding

---

## Advanced Evaluation (Optional)

For more detailed analysis, check:

1. **Regime transition matrix**: `reports/figures/regime_transitions.png`
2. **Feature distributions**: `reports/figures/feature_analysis.png`
3. **Volatility timeline**: `reports/figures/volatility_comparison.png`

Run comprehensive plotting:
```bash
python scripts/plot_regimes.py --plot-type all
```

---

## Summary: Your Action Items

Based on your current 2/5 score:

1. ✅ **COMPLETED:** Smoothed regimes with 10-day minimum
   - Result: Improved from 1/5 to 2/5
   - Location: `artifacts_smoothed_10day/`

2. ✅ **COMPLETED:** Retrained with 50-day rolling window
   - Result: Still 1/5 (no improvement)
   - With smoothing: 2/5
   - Location: `artifacts_rw50_smoothed_10day/`

3. ⏭️ **NEXT:** Try 3-state model for more granularity
   ```bash
   # Edit config to use n_states: 3
   python scripts/train_hmm.py --config configs/hmm_spx_studentt.yaml
   ```

4. ⏭️ **ALTERNATIVE:** Try different feature combinations
   - Add momentum indicators
   - Try exponential weighting (EWM)
   - Add volume-based features

5. ❌ **Do NOT:** Trade on current predictions (still too noisy and poorly separated)

---

## Questions?

- **"Why is my Cohen's d negative?"** → The model labeled regimes backwards (0=High, 1=Low instead of expected)
- **"Why so many transitions?"** → Rolling window is too small (20 days), making features noisy
- **"Is 100% confidence good?"** → Only if separation is also good! High confidence + poor separation = overconfident wrong predictions
- **"What's the minimum acceptable score?"** → 3/5 for research, 4/5 for live trading

Run evaluation anytime with:
```bash
python scripts/quick_evaluate.py
```

