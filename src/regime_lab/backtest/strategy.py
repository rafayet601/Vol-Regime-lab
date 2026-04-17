"""
Regime-conditioned trading strategy suite.

All strategies implement the interface:
    __call__(probs: ndarray, params: StudentTHMMParams) -> signal: ndarray

Convention (enforced by canonical label ordering in StudentTHMM):
    State 0 = LOW-volatility regime
    State 1, 2, ... = progressively higher-volatility regimes

FR-10: Four Strategy Implementations
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Threshold Strategy
# ---------------------------------------------------------------------------

class ThresholdStrategy:
    """
    Binary ±1 / 0 signal based on P(state 0) thresholds.

        signal_t = +1   if P(low-vol)_t > theta_high   (long)
        signal_t = -1   if P(low-vol)_t < theta_low    (short / hedge)
        signal_t =  0   otherwise                       (flat)

    Cleanest strategy for interview demo — directly interprets the
    model's state probability without continuous rescaling.
    """

    def __init__(
        self,
        theta_high: float = 0.65,
        theta_low:  float = 0.35,
        short_on_low: bool = False,
    ):
        """
        Parameters
        ----------
        theta_high    : P(low-vol) threshold for going long.
        theta_low     : P(low-vol) threshold below which we go short (if
                        short_on_low=True) or flat (if False).
        short_on_low  : If False, signal is 0 when below theta_low (safer).
        """
        self.theta_high   = theta_high
        self.theta_low    = theta_low
        self.short_on_low = short_on_low

    def __call__(self, probs: np.ndarray, params=None) -> np.ndarray:
        """
        Parameters
        ----------
        probs  : (T, K)  posterior state probabilities
        params : StudentTHMMParams (unused here but kept for uniform interface)

        Returns
        -------
        signal : (T,)  values in {-1, 0, +1}
        """
        p_low  = probs[:, 0]                   # P(state 0 = low-vol)
        signal = np.zeros(len(p_low))
        signal[p_low >  self.theta_high] =  1.0
        signal[p_low <  self.theta_low]  = -1.0 if self.short_on_low else 0.0
        return signal

    def __repr__(self):
        return (f"ThresholdStrategy(theta_high={self.theta_high}, "
                f"theta_low={self.theta_low}, short={self.short_on_low})")


# ---------------------------------------------------------------------------
# 2. Probability-Weighted Strategy
# ---------------------------------------------------------------------------

class ProbabilityWeightedStrategy:
    """
    Continuous signal proportional to regime conviction.

        signal_t = clip( 2 × P(low-vol)_t − 1,  −1,  +1 )

    Maps P(low-vol) ∈ [0,1] linearly to signal ∈ [-1, +1]:
        P=1.0 → +1.0  (fully long)
        P=0.5 →  0.0  (flat)
        P=0.0 → -1.0  (fully short)

    Avoids the discontinuity of threshold strategies and gives
    proportional sizing to model conviction.
    """

    def __call__(self, probs: np.ndarray, params=None) -> np.ndarray:
        p_low  = probs[:, 0]
        signal = np.clip(2.0 * p_low - 1.0, -1.0, 1.0)
        return signal

    def __repr__(self):
        return "ProbabilityWeightedStrategy()"


# ---------------------------------------------------------------------------
# 3. Vol-Targeting Strategy
# ---------------------------------------------------------------------------

class VolTargetingStrategy:
    """
    Position size scaled to target a constant realised volatility.

        position_t = vol_target / vol_forecast_t

    vol_forecast_t is regime-weighted realised vol from the HMM params:

        σ_forecast = Σ_k P(state k)_t × ||σ_k||₂ / sqrt(d)

    This is the standard Citadel / Millennium approach: maintain a
    constant ex-ante risk budget regardless of market regime.

    Parameters
    ----------
    vol_target : float
        Annualised volatility target (e.g. 0.10 = 10%).
    max_leverage : float
        Maximum allowed position size (default 2.0×).
    """

    def __init__(self, vol_target: float = 0.10, max_leverage: float = 2.0):
        self.vol_target   = vol_target
        self.max_leverage = max_leverage

    def __call__(self, probs: np.ndarray, params) -> np.ndarray:
        """
        Parameters
        ----------
        probs  : (T, K)
        params : StudentTHMMParams  — used for σ_k to compute vol forecast
        """
        K = probs.shape[1]
        d = params.sigma.shape[1]

        # Per-state annualised vol estimate: L2-norm of σ_k / sqrt(d) * sqrt(252)
        state_vols = np.linalg.norm(params.sigma, axis=1) / np.sqrt(d) * np.sqrt(252)

        # Regime-weighted vol forecast per time step
        vol_forecast = probs @ state_vols       # (T,)
        vol_forecast = np.maximum(vol_forecast, 1e-6)

        position = self.vol_target / vol_forecast
        position = np.clip(position, 0.0, self.max_leverage)

        return position

    def __repr__(self):
        return f"VolTargetingStrategy(vol_target={self.vol_target}, max_lev={self.max_leverage})"


# ---------------------------------------------------------------------------
# 4. Regime-Switching Strategy
# ---------------------------------------------------------------------------

class RegimeSwitchingStrategy:
    """
    Event-driven: go long for hold_period days after a high-to-low
    volatility regime transition.

    Regime transition detected when:
        Viterbi state switches from k > 0 (high-vol)  →  k = 0 (low-vol)

    Strategy logic
    --------------
    Day of transition detected → enter long position.
    Hold for hold_period days, then exit (signal = 0).
    No re-entry during an active hold.

    Parameters
    ----------
    hold_period : int
        Number of days to hold the position after a transition signal.
    """

    def __init__(self, hold_period: int = 5):
        self.hold_period = hold_period

    def __call__(self, probs: np.ndarray, params=None) -> np.ndarray:
        T = len(probs)
        states = np.argmax(probs, axis=1)      # MAP state at each step

        signal   = np.zeros(T)
        hold_cnt = 0

        for t in range(1, T):
            if states[t] == 0 and states[t - 1] > 0:
                # High-to-low transition detected
                hold_cnt = self.hold_period
            if hold_cnt > 0:
                signal[t] = 1.0
                hold_cnt -= 1

        return signal

    def __repr__(self):
        return f"RegimeSwitchingStrategy(hold_period={self.hold_period})"


# ---------------------------------------------------------------------------
# Strategy suite builder
# ---------------------------------------------------------------------------

def build_strategy_suite(
    theta_high:   float = 0.65,
    theta_low:    float = 0.35,
    vol_target:   float = 0.10,
    max_leverage: float = 2.0,
    hold_period:  int   = 5,
) -> Dict[str, object]:
    """
    Return a named dict of all four strategies for grid comparison.

    Usage
    -----
    suite = build_strategy_suite()
    for name, strat in suite.items():
        signal = strat(probs, params)
    """
    return {
        "threshold":      ThresholdStrategy(theta_high=theta_high, theta_low=theta_low),
        "prob_weighted":  ProbabilityWeightedStrategy(),
        "vol_targeting":  VolTargetingStrategy(vol_target=vol_target, max_leverage=max_leverage),
        "regime_switch":  RegimeSwitchingStrategy(hold_period=hold_period),
    }
