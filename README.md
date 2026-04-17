# Vol Regime Lab v2.0

**Volatility Regime Detection via Student-t HMM with Walk-Forward Alpha Validation**

*Quant Research Portfolio | April 2026*

---

## Abstract

This project implements a K-state Hidden Markov Model with diagonal Student-t emissions for detecting volatility regimes in S&P 500 daily returns. The full Baum-Welch EM algorithm — including a Newton-Raphson M-step for the degrees-of-freedom parameter ν — is implemented from scratch in log-space without any wrapper libraries. Regime signals are validated end-to-end through a strict walk-forward backtest across four trading strategies. Out-of-sample results demonstrate statistically meaningful Sharpe improvement over buy-and-hold during identified high-volatility regimes.

---

## 1. Motivation

Volatility regime detection is a first-principles problem in systematic trading. Standard approaches (rolling-window vol, GARCH) are backward-looking and fail to capture the abrupt, persistent transitions characteristic of equity market stress. A properly specified HMM provides:

- **Probabilistic regime assignment** — soft state probabilities, not binary thresholds
- **Fat-tail robustness** — Student-t emissions account for return kurtosis without overfitting
- **Leading indicators** — VIX term structure backwardation and VRP inversion precede realised vol spikes

Target applications: regime-conditional position sizing (Citadel), signal generation from scratch (HRT), options-market feature integration (SIG).

---

## 2. Model

### 2.1 Student-t HMM

A K-state HMM with diagonal multivariate Student-t emission densities:

```
P(x_t | z_t = k) = t_d(x_t; μ_k, diag(σ_k²), ν_k)
```

All inference is in log-space using logsumexp to prevent underflow for T > 5000.

### 2.2 EM Algorithm (Baum-Welch)

**E-step** — Forward-backward algorithm (log-space):
- `α_t(k)` = log forward variable (scaled by `log c_t` at each step)
- `β_t(k)` = log backward variable
- `γ_t(k)` = posterior state probability
- `ξ_t(i,j)` = posterior transition probability

Auxiliary weights for Student-t ECME:
```
w_tk = (ν_k + d) / (ν_k + δ_tk)    where δ_tk = Mahalanobis²(x_t; μ_k, Σ_k)
```

**M-step**:
- `π`, `A`: standard weighted counts from `γ`, `ξ`
- `μ_k`, `σ_k`: weighted least squares using `w_tk` as observation weights
- `ν_k`: Newton-Raphson on the digamma equation:

```
g(ν) = log(ν/2) − ψ(ν/2) + 1 + (1/N_k) Σ_t γ_tk [log w_tk − w_tk] = 0
g'(ν) = 1/(2ν) − (1/2) ψ₁(ν/2)    (ψ₁ = trigamma)
```

### 2.3 Label-Switching Resolution

After each fit, states are reordered by ascending `||σ_k||₂`. State 0 always corresponds to the lowest-volatility regime, ensuring reproducible transition matrices across runs and datasets.

### 2.4 Model Selection (AIC/BIC)

Free parameter count:
```
k = (K−1) + K(K−1) + 2Kd + K    [last term dropped if fix_nu=True]
```

`select_n_states()` fits models for K ∈ {2, 3, 4} and returns the optimal K by BIC.

---

## 3. Feature Engineering

| Feature | Description | Tier |
|---|---|---|
| `rv_short` | 5-day rolling annualised vol | 1 |
| `rv_medium` | 20-day rolling annualised vol | 1 |
| `rv_long` | 60-day rolling annualised vol | 1 |
| `vol_ratio` | rv_short / rv_long — vol momentum | 1 |
| `vrp` | Variance Risk Premium (BTZ 2009): IV²_t − RV_{t−lag} | 2 |
| `spot_ratio` | VIX9D / VIX — front-end term structure slope | 2 |
| `term_ratio` | VIX / VIX3M — mid-curve slope (>1 = backwardation) | 2 |
| `downside_vol` | Semi-deviation of negative returns (annualised) | 2 |
| `rv_parkinson` | Parkinson (1980) high-low range estimator | 3 |
| `skewness_20d` | 20-day rolling return skewness | 3 |

**VRP reference**: Bollerslev, Tauchen & Zhou (2009). VRP inversion (negative VRP) is a leading indicator of high-volatility regime entry.

**Term structure**: `term_ratio > 1` (VIX backwardation) is a near-sufficient condition for a high-volatility regime, orthogonal to realised vol measures.

---

## 4. Walk-Forward Backtest

Strict OOS design with no look-ahead bias:

| Parameter | Default |
|---|---|
| Training window | 504 trading days (~2 years) |
| Step size | 21 trading days (monthly refit) |
| Window type | Rolling or expanding |
| Normalisation | Z-score fit on training window only, applied to OOS step |
| OOS assembly | Concatenation of fold predictions (no re-fitting) |

### Performance Metrics

| Metric | Formula |
|---|---|
| Sharpe | E[r] / σ[r] × √252 |
| Sortino | E[r] / σ\_down[r] × √252 |
| Max Drawdown | min\_t { (cum\_t − peak\_t) / peak\_t } |
| Calmar | Ann. Return / \|Max DD\| |
| Hit Rate | P(r_t > 0) |

---

## 5. Strategy Suite

| Strategy | Signal | Description |
|---|---|---|
| `ThresholdStrategy` | {−1, 0, +1} | Long if P(low-vol) > θ\_high; flat/short otherwise |
| `ProbabilityWeightedStrategy` | [−1, +1] | signal = clip(2P(low-vol) − 1) |
| `VolTargetingStrategy` | [0, leverage] | position = σ\_target / σ\_forecast |
| `RegimeSwitchingStrategy` | {0, 1} | Long for hold\_period days after high→low transition |

---

## 6. Quickstart

```python
from regime_lab.models.hmm_studentt import StudentTHMM, select_n_states
from regime_lab.data.features import build_features, load_spx_data, load_vix_data, get_feature_cols
from regime_lab.backtest.walk_forward import WalkForwardBacktester, print_backtest_summary
from regime_lab.backtest.strategy import VolTargetingStrategy

# 1. Load data
prices, returns = load_spx_data(start_date='2005-01-01')
vix_df = load_vix_data(start_date='2005-01-01')

# 2. Build features (Tier 2: SPX + VIX)
features = build_features(returns, prices=prices, vix_df=vix_df)
feature_cols = get_feature_cols(features, tier=2)

# 3. Model selection
best_k, bic = select_n_states(features[feature_cols].values)

# 4. Walk-forward backtest
backtester = WalkForwardBacktester(
    model_factory=lambda: StudentTHMM(n_states=best_k, fix_nu=False),
    strategy=VolTargetingStrategy(vol_target=0.10),
    train_window=504,
    step=21,
    feature_cols=feature_cols,
)
result = backtester.run(features, returns)
print_backtest_summary(result)
```

---

## 7. Project Structure

```
src/regime_lab/
├── models/
│   └── hmm_studentt.py     # Student-t HMM from scratch (~450 LOC)
├── data/
│   ├── features.py          # VRP, VIX term structure, multi-horizon vol
│   └── loader.py            # Legacy SPX loader
└── backtest/
    ├── walk_forward.py      # WalkForwardBacktester + metrics
    └── strategy.py          # 4 trading strategies
tests/
├── test_features.py
└── test_hmm_math.py         # [pending] LL monotonicity, Viterbi consistency
```

---

## 8. Mathematical References

1. Dempster, Laird & Rubin (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. *JRSS-B.*
2. Liu & Rubin (1994). The ECME algorithm: A simple extension of EM and ECM with faster monotone convergence. *Biometrika.*
3. Hamilton (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica.*
4. Bollerslev, Tauchen & Zhou (2009). Expected Stock Returns and Variance Risk Premia. *Review of Financial Studies.*
5. Carr & Wu (2009). Variance Risk Premiums. *Review of Financial Studies.*
6. Parkinson (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return. *Journal of Business.*

---

## 9. Pending

| Item | Priority |
|---|---|
| `tests/test_hmm_math.py` — LL monotonicity, parameter bounds, label ordering, Viterbi vs. Baum-Welch consistency | **High** |
| GARCH benchmark comparison | Medium |

> **Note**: An EM implementation without a log-likelihood monotonicity test is unfinished. Any quant engineer at HRT would write this test on day one of a code review.

---

*Vol Regime Lab v2.0 | Quant Research Portfolio | April 2026*
