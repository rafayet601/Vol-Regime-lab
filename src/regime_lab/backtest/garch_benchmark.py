"""
GARCH benchmark comparison for Vol Regime Lab v2.0.

Fits a GARCH(1,1) model on the same walk-forward schedule as the HMM,
generates a vol-targeting signal from GARCH conditional volatility, and
computes the same metric suite so results are directly comparable.

Usage
-----
from regime_lab.backtest.garch_benchmark import GARCHBenchmark, run_garch_benchmark

result = run_garch_benchmark(returns, train_window=504, step=21, vol_target=0.10)
print_garch_comparison(hmm_result, result)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container  (mirrors WalkForwardResult interface)
# ---------------------------------------------------------------------------

@dataclass
class GARCHBenchmarkResult:
    """Walk-forward GARCH(1,1) backtest output."""
    oos_returns:       pd.Series
    vol_forecast:      pd.Series          # one-step conditional vol (annualised)
    fold_metrics:      List[Dict]
    overall_metrics:   Dict[str, float]
    benchmark_metrics: Dict[str, float]   # buy-and-hold over same OOS period
    n_folds:           int = 0
    train_window:      int = 504
    step:              int = 21
    model_spec:        str = "GARCH(1,1)-Normal"


# ---------------------------------------------------------------------------
# GARCH walk-forward backtester
# ---------------------------------------------------------------------------

class GARCHBenchmark:
    """
    Walk-forward GARCH(1,1) benchmark.

    Identical schedule to WalkForwardBacktester (same train_window, step,
    grow_window) so output metrics are directly comparable.

    Strategy: vol-targeting
        position_t = vol_target / garch_vol_forecast_t
        position is clipped to [0, max_leverage]

    Parameters
    ----------
    train_window : int
        Days in each training window (default 504).
    step : int
        Days between refits (default 21).
    vol_target : float
        Annualised vol target (default 0.10 = 10 %).
    max_leverage : float
        Max allowed position size (default 2.0).
    grow_window : bool
        Expanding (True) or rolling (False) window.
    p, q : int
        GARCH lag orders (default 1, 1).
    dist : str
        Innovation distribution: 'normal', 't', 'skewt'.
    """

    def __init__(
        self,
        train_window:  int   = 504,
        step:          int   = 21,
        vol_target:    float = 0.10,
        max_leverage:  float = 2.0,
        grow_window:   bool  = False,
        p:             int   = 1,
        q:             int   = 1,
        dist:          str   = "normal",
    ):
        self.train_window = train_window
        self.step         = step
        self.vol_target   = vol_target
        self.max_leverage = max_leverage
        self.grow_window  = grow_window
        self.p            = p
        self.q            = q
        self.dist         = dist

    def run(self, returns: pd.Series) -> GARCHBenchmarkResult:
        """
        Execute the walk-forward GARCH backtest.

        Parameters
        ----------
        returns : pd.Series of daily log returns.

        Returns
        -------
        GARCHBenchmarkResult
        """
        from arch import arch_model
        from regime_lab.backtest.walk_forward import compute_metrics

        r_all = returns.values
        idx   = returns.index
        N     = len(r_all)

        if N < self.train_window + self.step:
            raise ValueError(
                f"Not enough data: N={N} < train_window+step={self.train_window+self.step}"
            )

        oos_ret_list:  List[pd.Series] = []
        vol_fc_list:   List[pd.Series] = []
        fold_metrics:  List[Dict]      = []

        fold = 0
        t    = self.train_window

        while t < N:
            oos_end = min(t + self.step, N)

            train_start = 0 if self.grow_window else max(0, t - self.train_window)
            r_train = r_all[train_start:t]

            # ----------------------------------------------------------------
            # Fit GARCH(p,q) on training window
            # ----------------------------------------------------------------
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    am = arch_model(
                        r_train * 100,      # scale for numerical stability
                        vol="Garch",
                        p=self.p,
                        q=self.q,
                        dist=self.dist,
                        rescale=False,
                    )
                    res = am.fit(disp="off", show_warning=False)
            except Exception as exc:
                logger.warning(f"Fold {fold}: GARCH fit failed ({exc}), skipping.")
                t += self.step
                fold += 1
                continue

            # ----------------------------------------------------------------
            # One-step-ahead vol forecast for each OOS day
            # ----------------------------------------------------------------
            r_oos    = r_all[t:oos_end]
            idx_oos  = idx[t:oos_end]
            T_oos    = len(r_oos)

            vol_forecast = np.zeros(T_oos)

            # We refit at the fold boundary; for simplicity we use the
            # last conditional variance from training as the seed and
            # recursively propagate using the GARCH recurrence.
            omega = float(res.params.get("omega", 1e-6))
            alpha = float(res.params.get("alpha[1]", 0.1))
            beta  = float(res.params.get("beta[1]",  0.85))

            # Last in-sample conditional variance (scaled ×100²)
            h_prev = float(res.conditional_volatility[-1] ** 2)
            eps_prev = float((r_train[-1] * 100) ** 2)

            for i in range(T_oos):
                h_t = omega + alpha * eps_prev + beta * h_prev
                # Annualised vol: sqrt(h_t) / 100 * sqrt(252)
                vol_forecast[i] = np.sqrt(h_t) / 100.0 * np.sqrt(252)
                # Update for next step using OOS return (simulated OOS)
                eps_prev = (r_oos[i] * 100) ** 2
                h_prev   = h_t

            vol_forecast = np.maximum(vol_forecast, 1e-6)

            # ----------------------------------------------------------------
            # Vol-targeting signal
            # ----------------------------------------------------------------
            position  = np.clip(self.vol_target / vol_forecast, 0.0, self.max_leverage)
            strat_ret = r_oos * position

            oos_ret_list.append(pd.Series(strat_ret,    index=idx_oos, name="garch_strategy"))
            vol_fc_list.append(pd.Series(vol_forecast, index=idx_oos, name="garch_vol"))
            fold_metrics.append(compute_metrics(strat_ret))

            logger.info(
                f"Fold {fold:3d}  train=[{train_start},{t})  oos=[{t},{oos_end})"
                f"  sharpe={fold_metrics[-1]['sharpe']:.2f}"
            )

            t += self.step
            fold += 1

        if not oos_ret_list:
            raise RuntimeError("No OOS folds completed.")

        oos_returns  = pd.concat(oos_ret_list)
        vol_forecast = pd.concat(vol_fc_list)

        bh_ret            = returns.loc[oos_returns.index].values
        benchmark_metrics = compute_metrics(bh_ret)
        overall_metrics   = compute_metrics(oos_returns.values)

        return GARCHBenchmarkResult(
            oos_returns       = oos_returns,
            vol_forecast      = vol_forecast,
            fold_metrics      = fold_metrics,
            overall_metrics   = overall_metrics,
            benchmark_metrics = benchmark_metrics,
            n_folds           = fold,
            train_window      = self.train_window,
            step              = self.step,
            model_spec        = f"GARCH({self.p},{self.q})-{self.dist}",
        )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_garch_benchmark(
    returns:      pd.Series,
    train_window: int   = 504,
    step:         int   = 21,
    vol_target:   float = 0.10,
    max_leverage: float = 2.0,
    grow_window:  bool  = False,
) -> GARCHBenchmarkResult:
    """
    Fit and run a GARCH(1,1) walk-forward benchmark.

    Parameters
    ----------
    returns : daily log return series.
    (remaining params match WalkForwardBacktester defaults)

    Returns
    -------
    GARCHBenchmarkResult
    """
    bench = GARCHBenchmark(
        train_window=train_window,
        step=step,
        vol_target=vol_target,
        max_leverage=max_leverage,
        grow_window=grow_window,
    )
    return bench.run(returns)


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def print_garch_comparison(
    hmm_result,
    garch_result: GARCHBenchmarkResult,
) -> None:
    """
    Print a side-by-side metric table: HMM strategy vs GARCH vs buy-and-hold.

    Parameters
    ----------
    hmm_result   : WalkForwardResult from WalkForwardBacktester.run()
    garch_result : GARCHBenchmarkResult from GARCHBenchmark.run()
    """
    metrics = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate"]
    labels  = ["Ann Return", "Ann Vol",  "Sharpe", "Sortino", "Max DD",       "Calmar", "Hit Rate"]

    col_w = 14
    header = (
        f"{'Metric':<16}"
        f"{'HMM (OOS)':>{col_w}}"
        f"{'GARCH (OOS)':>{col_w}}"
        f"{'Buy & Hold':>{col_w}}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"  HMM vs GARCH({garch_result.model_spec}) Walk-Forward Comparison")
    print(f"  HMM folds: {hmm_result.n_folds}   "
          f"GARCH folds: {garch_result.n_folds}   "
          f"Train: {garch_result.train_window}d   Step: {garch_result.step}d")
    print(sep)
    print(header)
    print(sep)

    for m, lbl in zip(metrics, labels):
        hmm_v  = hmm_result.overall_metrics.get(m, np.nan)
        gch_v  = garch_result.overall_metrics.get(m, np.nan)
        bh_v   = garch_result.benchmark_metrics.get(m, np.nan)

        if m in ("ann_return", "ann_vol", "max_drawdown", "hit_rate"):
            fmt = "{:.1%}"
        else:
            fmt = "{:.2f}"

        print(
            f"{lbl:<16}"
            f"{fmt.format(hmm_v):>{col_w}}"
            f"{fmt.format(gch_v):>{col_w}}"
            f"{fmt.format(bh_v):>{col_w}}"
        )

    print(sep + "\n")
