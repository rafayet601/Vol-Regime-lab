"""
Walk-forward backtester with strict look-ahead bias prevention.

Design
------
- Model at time t is trained exclusively on data with index < t.
- OOS fold predictions are assembled by concatenation only —
  no re-fitting on the assembled sequence.
- Feature normalisation (z-scoring) is fit inside each training window
  and applied to the corresponding OOS step.

FR-08: Strict Look-Ahead Bias Prevention
FR-09: Performance Metrics (Sharpe, Sortino, MDD, Calmar, Hit Rate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performance metrics  (FR-09)
# ---------------------------------------------------------------------------

def _sharpe(r: np.ndarray) -> float:
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _sortino(r: np.ndarray) -> float:
    neg = r[r < 0]
    if len(neg) == 0 or neg.std() == 0:
        return 0.0
    return float(r.mean() / neg.std() * np.sqrt(252))


def _max_drawdown(r: np.ndarray) -> float:
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def _calmar(r: np.ndarray) -> float:
    ann_ret = float(r.mean() * 252)
    mdd = abs(_max_drawdown(r))
    return ann_ret / mdd if mdd > 1e-10 else 0.0


def _hit_rate(r: np.ndarray) -> float:
    return float((r > 0).mean())


def compute_metrics(r: np.ndarray) -> Dict[str, float]:
    """
    Full metric suite for a return series (daily, log or arithmetic).

    Returns
    -------
    dict with keys: sharpe, sortino, max_drawdown, calmar,
                    hit_rate, ann_return, ann_vol
    """
    r = np.asarray(r, dtype=float)
    return {
        "sharpe":       _sharpe(r),
        "sortino":      _sortino(r),
        "max_drawdown": _max_drawdown(r),
        "calmar":       _calmar(r),
        "hit_rate":     _hit_rate(r),
        "ann_return":   float(r.mean() * 252),
        "ann_vol":      float(r.std() * np.sqrt(252)),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """All outputs from a walk-forward backtest run."""
    oos_returns:       pd.Series               # strategy daily returns (OOS)
    regime_probs:      pd.DataFrame            # P(state k) per day (OOS)
    predicted_states:  pd.Series               # Viterbi MAP state (OOS)
    fold_metrics:      List[Dict[str, float]]  # per-fold OOS metrics
    overall_metrics:   Dict[str, float]        # aggregate OOS metrics
    benchmark_metrics: Dict[str, float]        # buy-and-hold over same OOS period
    is_metrics:        Dict[str, float]        # last-fold IS metrics
    n_folds:           int = 0
    train_window:      int = 504
    step:              int = 21


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Strict walk-forward backtester.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a fresh (unfitted) StudentTHMM.
    strategy : callable
        Implements __call__(probs: ndarray, params) -> signal: ndarray
        where probs is (T_oos, K) and signal is (T_oos,).
    train_window : int
        Number of trading days in each training window (default 504 ≈ 2yr).
    step : int
        Number of days between refits (default 21 ≈ monthly).
    grow_window : bool
        If True, training window expands from `train_window` to t (expanding).
        If False, strictly rolling window of fixed `train_window` days.
    feature_cols : list of str or None
        Feature columns to use.  If None, all columns except 'returns' are used.
    normalize : bool
        If True, z-score features using train-window mean/std before fitting.
    """

    def __init__(
        self,
        model_factory: Callable,
        strategy: Callable,
        train_window: int = 504,
        step: int = 21,
        grow_window: bool = False,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
    ):
        self.model_factory = model_factory
        self.strategy = strategy
        self.train_window = train_window
        self.step = step
        self.grow_window = grow_window
        self.feature_cols = feature_cols
        self.normalize = normalize

    # ------------------------------------------------------------------

    def run(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
    ) -> WalkForwardResult:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        features : DataFrame produced by build_features() — must include a
                   'returns' column and all feature columns.
        returns  : raw SPX returns aligned to features.index (used for
                   benchmark computation and strategy PnL calculation).

        Returns
        -------
        WalkForwardResult
        """
        # Align returns and features on a common index
        common_idx = features.index.intersection(returns.index)
        features = features.loc[common_idx]
        returns  = returns.loc[common_idx]

        # Determine feature columns
        fcols = self.feature_cols
        if fcols is None:
            fcols = [c for c in features.columns if c != "returns"]

        X_all = features[fcols].values         # (N, d)
        r_all = returns.values                  # (N,)
        idx   = features.index
        N     = len(X_all)

        if N < self.train_window + self.step:
            raise ValueError(
                f"Not enough data: N={N} < train_window+step={self.train_window+self.step}"
            )

        # Storage
        oos_returns_list:      List[pd.Series]     = []
        regime_probs_list:     List[pd.DataFrame]  = []
        predicted_states_list: List[pd.Series]     = []
        fold_metrics:          List[Dict]           = []
        last_is_metrics:       Dict[str, float]     = {}

        fold = 0
        t = self.train_window   # first OOS start index

        while t < N:
            oos_end = min(t + self.step, N)

            # --- Training slice (strictly < t) ---
            if self.grow_window:
                train_start = 0
            else:
                train_start = max(0, t - self.train_window)
            train_slice = slice(train_start, t)

            X_train = X_all[train_slice]
            r_train = r_all[train_slice]

            # Normalisation: fit scaler on training window only
            if self.normalize:
                mu_tr  = X_train.mean(axis=0)
                std_tr = X_train.std(axis=0)
                std_tr = np.where(std_tr < 1e-10, 1.0, std_tr)
                X_train_sc = (X_train - mu_tr) / std_tr
            else:
                mu_tr, std_tr = None, None
                X_train_sc = X_train

            # --- Fit model on training window ---
            model = self.model_factory()
            try:
                model.fit(X_train_sc)
            except Exception as exc:
                logger.warning(f"Fold {fold}: fit failed ({exc}), skipping.")
                t += self.step
                fold += 1
                continue

            params = model.params_

            # --- IS metrics (last fold) ---
            try:
                is_probs  = model.predict_proba(X_train_sc)
                is_signal = self.strategy(is_probs, params)
                is_ret    = r_train * is_signal[: len(r_train)]
                last_is_metrics = compute_metrics(is_ret)
            except Exception:
                pass

            # --- OOS slice ---
            X_oos = X_all[t:oos_end]
            r_oos = r_all[t:oos_end]
            idx_oos = idx[t:oos_end]

            if self.normalize:
                X_oos_sc = (X_oos - mu_tr) / std_tr
            else:
                X_oos_sc = X_oos

            # Predictions on OOS slice (no fitting on OOS data)
            try:
                probs_oos  = model.predict_proba(X_oos_sc)
                states_oos = model.predict(X_oos_sc)
                signal_oos = self.strategy(probs_oos, params)
            except Exception as exc:
                logger.warning(f"Fold {fold}: predict failed ({exc}), skipping.")
                t += self.step
                fold += 1
                continue

            # Strategy returns = signal × underlying return
            strat_ret = r_oos * signal_oos

            # Store
            oos_returns_list.append(
                pd.Series(strat_ret, index=idx_oos, name="strategy")
            )
            regime_probs_list.append(
                pd.DataFrame(
                    probs_oos,
                    index=idx_oos,
                    columns=[f"p_state_{k}" for k in range(probs_oos.shape[1])],
                )
            )
            predicted_states_list.append(
                pd.Series(states_oos, index=idx_oos, name="state")
            )

            fold_metrics.append(compute_metrics(strat_ret))

            logger.info(
                f"Fold {fold:3d}  train=[{train_start},{t})  oos=[{t},{oos_end})"
                f"  sharpe={fold_metrics[-1]['sharpe']:.2f}"
            )

            t += self.step
            fold += 1

        if not oos_returns_list:
            raise RuntimeError("No OOS folds completed — check data length and parameters.")

        # Assemble OOS outputs
        oos_returns      = pd.concat(oos_returns_list)
        regime_probs     = pd.concat(regime_probs_list)
        predicted_states = pd.concat(predicted_states_list)

        # Buy-and-hold benchmark over the same OOS period
        bh_ret = returns.loc[oos_returns.index].values
        benchmark_metrics = compute_metrics(bh_ret)

        overall_metrics = compute_metrics(oos_returns.values)

        return WalkForwardResult(
            oos_returns       = oos_returns,
            regime_probs      = regime_probs,
            predicted_states  = predicted_states,
            fold_metrics      = fold_metrics,
            overall_metrics   = overall_metrics,
            benchmark_metrics = benchmark_metrics,
            is_metrics        = last_is_metrics,
            n_folds           = fold,
            train_window      = self.train_window,
            step              = self.step,
        )


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_backtest_summary(result: WalkForwardResult) -> None:
    """Print a formatted comparison table of OOS, IS, and benchmark metrics."""
    col_w = 14
    metrics = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate"]
    labels  = ["Ann Return", "Ann Vol",  "Sharpe", "Sortino", "Max DD",       "Calmar", "Hit Rate"]

    header = f"{'Metric':<16}" + f"{'OOS Strategy':>{col_w}}" + f"{'IS (last)':>{col_w}}" + f"{'Buy & Hold':>{col_w}}"
    sep    = "-" * len(header)

    print("\n" + sep)
    print("  Walk-Forward Backtest Summary")
    print(f"  Folds: {result.n_folds}   Train: {result.train_window}d   Step: {result.step}d")
    print(sep)
    print(header)
    print(sep)

    for m, lbl in zip(metrics, labels):
        oos_v  = result.overall_metrics.get(m, np.nan)
        is_v   = result.is_metrics.get(m, np.nan)
        bh_v   = result.benchmark_metrics.get(m, np.nan)

        # Format as percent for return/vol/dd metrics
        if m in ("ann_return", "ann_vol", "max_drawdown"):
            fmt = "{:.1%}"
        elif m == "hit_rate":
            fmt = "{:.1%}"
        else:
            fmt = "{:.2f}"

        row = (
            f"{lbl:<16}"
            + f"{fmt.format(oos_v):>{col_w}}"
            + f"{fmt.format(is_v):>{col_w}}"
            + f"{fmt.format(bh_v):>{col_w}}"
        )
        print(row)

    print(sep + "\n")
