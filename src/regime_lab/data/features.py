"""
Feature engineering for Vol Regime Lab v2.0.

Feature tiers
-------------
Tier 1  (SPX only, 4 features):
    rv_short, rv_medium, rv_long, vol_ratio

Tier 2  (+ VIX, 8 features):
    Tier 1 + vrp, spot_ratio, term_ratio, downside_vol

Tier 3  (full, 12 features):
    Tier 2 + rv_parkinson, skewness_20d, <reserved>, <reserved>

References
----------
Bollerslev, Tauchen & Zhou (2009). Expected Stock Returns and Variance
    Risk Premia. Review of Financial Studies.
Parkinson (1980). The Extreme Value Method for Estimating the Variance
    of the Rate of Return. Journal of Business.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_spx_data(
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Download SPX (^GSPC) daily OHLCV and compute log returns.

    Returns
    -------
    prices  : DataFrame with columns [open, high, low, close, volume]
    returns : Series of daily log returns (NaN first row dropped)
    """
    import yfinance as yf

    ticker = yf.Ticker("^GSPC")
    prices = ticker.history(start=start_date, end=end_date)
    prices.columns = [c.lower() for c in prices.columns]
    prices.index = prices.index.tz_localize(None)

    returns = np.log(prices["close"] / prices["close"].shift(1)).dropna()
    returns.name = "returns"

    logger.info(f"Loaded SPX: {len(prices)} days  [{prices.index[0].date()} → {prices.index[-1].date()}]")
    return prices, returns


def load_vix_data(
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download VIX (^VIX), VIX9D (^VIX9D), VIX3M (^VIX3M) closing prices.

    Returns
    -------
    DataFrame with columns [vix, vix9d, vix3m] on a common date index.
    Missing tickers are filled with NaN (^VIX9D history is limited pre-2014).
    """
    import yfinance as yf

    tickers = {"vix": "^VIX", "vix9d": "^VIX9D", "vix3m": "^VIX3M"}
    frames = {}
    for col, sym in tickers.items():
        try:
            raw = yf.Ticker(sym).history(start=start_date, end=end_date)
            raw.index = raw.index.tz_localize(None)
            frames[col] = raw["Close"].rename(col)
        except Exception as exc:
            logger.warning(f"Could not download {sym}: {exc}")
            frames[col] = pd.Series(dtype=float, name=col)

    df = pd.concat(frames.values(), axis=1)
    df.columns = list(frames.keys())
    logger.info(f"Loaded VIX data: {len(df)} days")
    return df


# ---------------------------------------------------------------------------
# Realised variance estimators
# ---------------------------------------------------------------------------

def realized_variance_cc(returns: pd.Series, window: int) -> pd.Series:
    """
    Close-to-close realised variance (annualised):
        RV_t = 252 × rolling_mean( r_t² , window )

    This is the standard BTZ (2009) denominator.
    """
    rv = (returns ** 2).rolling(window).mean() * 252
    rv.name = f"rv_cc_{window}d"
    return rv


def realized_variance_parkinson(
    prices: pd.DataFrame, window: int = 20
) -> pd.Series:
    """
    Parkinson (1980) high-low range estimator (annualised):
        RV_pk = 252 / (4 ln 2)  × rolling_mean( (ln H/L)² , window )

    Approximately 5× more efficient than close-to-close for GBM.
    Requires columns 'high' and 'low'.
    """
    if "high" not in prices.columns or "low" not in prices.columns:
        logger.warning("Parkinson estimator requires high/low columns — returning NaN series.")
        return pd.Series(np.nan, index=prices.index, name="rv_parkinson")

    log_hl = np.log(prices["high"] / prices["low"])
    rv_pk = (log_hl ** 2).rolling(window).mean() * 252 / (4.0 * np.log(2))
    rv_pk.name = "rv_parkinson"
    return rv_pk


# ---------------------------------------------------------------------------
# VRP  (FR-04)
# ---------------------------------------------------------------------------

def variance_risk_premium(
    returns: pd.Series,
    vix: pd.Series,
    rv_window: int = 20,
    lag: int = 0,
) -> pd.Series:
    """
    Variance Risk Premium  (Bollerslev, Tauchen & Zhou 2009):

        VRP_t = IV²_t − RV_{t-lag}

    where
        IV²_t  = (VIX_t / 100)² × 252   — annualised implied variance
        RV_t   = close-to-close realised variance (rv_window-day rolling)

    A positive VRP means investors pay a premium for variance insurance.
    VRP sign reversal (going negative) is a leading indicator of a
    high-volatility regime.

    Parameters
    ----------
    lag : int
        Lag for RV relative to IV (default 0 = contemporaneous).
    """
    iv2 = (vix / 100.0) ** 2 * 252
    rv  = realized_variance_cc(returns, rv_window)

    # Align on common index
    common = iv2.index.intersection(rv.index)
    iv2 = iv2.loc[common]
    rv  = rv.loc[common]

    if lag > 0:
        rv = rv.shift(lag)

    vrp = iv2 - rv
    vrp.name = "vrp"
    return vrp


# ---------------------------------------------------------------------------
# VIX term structure  (FR-05)
# ---------------------------------------------------------------------------

def vix_term_structure_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct VIX term structure slope features.

        spot_ratio = VIX9D / VIX    (front-end slope; <1 = contango)
        term_ratio = VIX   / VIX3M  (mid-curve;  >1 = backwardation → high-vol)

    term_ratio > 1 is a near-sufficient condition for a high-volatility regime.
    """
    out = pd.DataFrame(index=vix_df.index)

    vix   = vix_df.get("vix")
    vix9d = vix_df.get("vix9d")
    vix3m = vix_df.get("vix3m")

    if vix is not None and vix9d is not None:
        out["spot_ratio"] = (vix9d / vix.replace(0, np.nan)).clip(0.3, 3.0)
    else:
        out["spot_ratio"] = np.nan

    if vix is not None and vix3m is not None:
        out["term_ratio"] = (vix / vix3m.replace(0, np.nan)).clip(0.3, 3.0)
    else:
        out["term_ratio"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Multi-horizon vol  (FR-06)
# ---------------------------------------------------------------------------

def multi_horizon_vol(returns: pd.Series) -> pd.DataFrame:
    """
    Rolling standard deviation at three horizons (annualised):
        rv_short  = 5-day
        rv_medium = 20-day
        rv_long   = 60-day
        vol_ratio = rv_short / rv_long  (>1 = vol accelerating)
    """
    rv_short  = returns.rolling(5).std()  * np.sqrt(252)
    rv_medium = returns.rolling(20).std() * np.sqrt(252)
    rv_long   = returns.rolling(60).std() * np.sqrt(252)

    vol_ratio = (rv_short / rv_long.replace(0, np.nan)).clip(0.1, 10.0)

    return pd.DataFrame({
        "rv_short":  rv_short,
        "rv_medium": rv_medium,
        "rv_long":   rv_long,
        "vol_ratio": vol_ratio,
    }, index=returns.index)


# ---------------------------------------------------------------------------
# Downside risk & skewness  (FR-07)
# ---------------------------------------------------------------------------

def downside_skew_features(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Semi-deviation of negative returns (downside_vol) and rolling skewness.

        downside_vol  = std of returns[r < 0] over rolling window (annualised)
        skewness_20d  = rolling 20-day skewness of returns

    Negative skew regimes are high-stress, high-drawdown environments.
    """
    downside_vol = (
        returns
        .where(returns < 0, other=np.nan)
        .rolling(window, min_periods=5)
        .std()
        * np.sqrt(252)
    )
    downside_vol.name = "downside_vol"

    skewness_20d = returns.rolling(window, min_periods=5).skew()
    skewness_20d.name = "skewness_20d"

    return pd.DataFrame({"downside_vol": downside_vol, "skewness_20d": skewness_20d})


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(
    returns: pd.Series,
    prices: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix.

    Parameters
    ----------
    returns : pd.Series  — daily log returns (SPX)
    prices  : pd.DataFrame — OHLCV data (needed for Parkinson; optional)
    vix_df  : pd.DataFrame — output of load_vix_data() (optional)

    Returns
    -------
    DataFrame with all computed features and the returns column,
    with leading NaN rows dropped.

    Feature columns (tiers)
    -----------------------
    Tier 1 (always):   rv_short, rv_medium, rv_long, vol_ratio
    Tier 2 (+vix_df):  + vrp, spot_ratio, term_ratio, downside_vol
    Tier 3 (+prices):  + rv_parkinson, skewness_20d
    """
    frames: list[pd.DataFrame | pd.Series] = [returns.rename("returns")]

    # --- Tier 1: multi-horizon vol ---
    mv = multi_horizon_vol(returns)
    frames.append(mv)

    # --- Tier 2: VRP + term structure + downside ---
    if vix_df is not None:
        vix_aligned = vix_df.reindex(returns.index).ffill()

        # VRP
        if "vix" in vix_aligned.columns:
            vrp = variance_risk_premium(returns, vix_aligned["vix"])
            frames.append(vrp.rename("vrp"))

        # Term structure
        ts = vix_term_structure_features(vix_aligned)
        frames.append(ts)

        # Downside vol
        ds = downside_skew_features(returns, window=20)
        frames.append(ds[["downside_vol"]])

    # --- Tier 3: Parkinson + skewness ---
    if prices is not None:
        prices_aligned = prices.reindex(returns.index)
        rv_pk = realized_variance_parkinson(prices_aligned, window=20)
        frames.append(rv_pk)

    # Skewness is cheap, always include if vix_df present (Tier 2+)
    if vix_df is not None:
        sk = downside_skew_features(returns, window=20)
        frames.append(sk[["skewness_20d"]])

    df = pd.concat(frames, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        logger.info(f"build_features: dropped {dropped} rows with NaN/Inf")

    logger.info(f"build_features: {len(df)} rows × {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Feature column selector by tier
# ---------------------------------------------------------------------------

_TIER1_COLS  = ["rv_short", "rv_medium", "rv_long", "vol_ratio"]
_TIER2_COLS  = _TIER1_COLS + ["vrp", "spot_ratio", "term_ratio", "downside_vol"]
_TIER3_COLS  = _TIER2_COLS + ["rv_parkinson", "skewness_20d"]


def get_feature_cols(df: pd.DataFrame, tier: int = 2) -> List[str]:
    """
    Return the list of feature columns available in df for the requested tier.

    Columns missing from df (e.g. VIX not loaded) are silently excluded.
    """
    if tier == 1:
        candidates = _TIER1_COLS
    elif tier == 2:
        candidates = _TIER2_COLS
    else:
        candidates = _TIER3_COLS

    available = [c for c in candidates if c in df.columns]
    logger.info(f"get_feature_cols: tier={tier}  available={available}")
    return available
