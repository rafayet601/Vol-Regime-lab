"""
Tests for v2.0 feature engineering (features.py).

Covers all functional requirements:
  FR-04: Variance Risk Premium
  FR-05: VIX term structure features
  FR-06: Multi-horizon volatility
  FR-07: Downside risk & skewness
  build_features / get_feature_cols tier system
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.data.features import (
    build_features,
    get_feature_cols,
    variance_risk_premium,
    vix_term_structure_features,
    realized_variance_cc,
    realized_variance_parkinson,
    multi_horizon_vol,
    downside_skew_features,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def returns_500():
    """500 business-day synthetic log-return series."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=500, freq="B")
    r = pd.Series(rng.normal(0, 0.01, 500), index=dates, name="returns")
    return r


@pytest.fixture(scope="module")
def vix_df_500(returns_500):
    """Synthetic VIX / VIX9D / VIX3M aligned to returns_500."""
    rng = np.random.default_rng(1)
    idx = returns_500.index
    vix   = pd.Series(np.abs(rng.normal(18, 4, len(idx))), index=idx, name="vix")
    vix9d = vix * rng.uniform(0.90, 1.10, len(idx))
    vix3m = vix * rng.uniform(0.90, 1.10, len(idx))
    return pd.DataFrame({"vix": vix, "vix9d": vix9d, "vix3m": vix3m})


@pytest.fixture(scope="module")
def prices_500(returns_500):
    """Synthetic OHLCV prices compatible with Parkinson estimator."""
    rng = np.random.default_rng(2)
    idx = returns_500.index
    close = 100 * np.cumprod(1 + returns_500.values)
    high  = close * (1 + np.abs(rng.normal(0, 0.005, len(idx))))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, len(idx))))
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)


# ---------------------------------------------------------------------------
# FR-06: Multi-horizon volatility
# ---------------------------------------------------------------------------

class TestMultiHorizonVol:

    def test_columns_present(self, returns_500):
        mv = multi_horizon_vol(returns_500)
        for col in ["rv_short", "rv_medium", "rv_long", "vol_ratio"]:
            assert col in mv.columns, f"Missing column: {col}"

    def test_rv_ordering(self, returns_500):
        """On any given day, rv_short should exhibit higher variance than rv_long
        (short-window estimates are noisier). Check mean levels are plausible."""
        mv = multi_horizon_vol(returns_500).dropna()
        # rv_long is a smoothed estimate — its values should span a narrower range
        assert mv["rv_long"].std() < mv["rv_short"].std(), (
            "rv_long should be smoother (lower std) than rv_short"
        )

    def test_vol_ratio_positive(self, returns_500):
        mv = multi_horizon_vol(returns_500).dropna()
        assert (mv["vol_ratio"] > 0).all(), "vol_ratio must be positive"

    def test_vol_ratio_clipped(self, returns_500):
        mv = multi_horizon_vol(returns_500).dropna()
        assert (mv["vol_ratio"] <= 10.0).all(), "vol_ratio should be clipped at 10"
        assert (mv["vol_ratio"] >= 0.1).all(), "vol_ratio should be clipped at 0.1"

    def test_no_nan_after_warm_up(self, returns_500):
        """After the 60-day warm-up, no NaN should remain."""
        mv = multi_horizon_vol(returns_500).iloc[60:]
        assert not mv.isnull().any().any(), "NaN found after 60-day warm-up"

    def test_annualisation(self, returns_500):
        """rv_medium (20d std × √252) should be in a plausible annual vol range."""
        mv = multi_horizon_vol(returns_500).dropna()
        # Synthetic data has σ=0.01 daily → ~16% annual
        mean_rv = mv["rv_medium"].mean()
        assert 0.05 < mean_rv < 0.50, f"rv_medium mean {mean_rv:.3f} out of plausible range"


# ---------------------------------------------------------------------------
# FR-04: Variance Risk Premium
# ---------------------------------------------------------------------------

class TestVarianceRiskPremium:

    def test_vrp_is_series(self, returns_500, vix_df_500):
        vrp = variance_risk_premium(returns_500, vix_df_500["vix"])
        assert isinstance(vrp, pd.Series)

    def test_vrp_name(self, returns_500, vix_df_500):
        vrp = variance_risk_premium(returns_500, vix_df_500["vix"])
        assert vrp.name == "vrp"

    def test_vrp_finite_after_dropna(self, returns_500, vix_df_500):
        vrp = variance_risk_premium(returns_500, vix_df_500["vix"]).dropna()
        assert np.all(np.isfinite(vrp.values)), "VRP contains non-finite values"
        assert len(vrp) > 0, "VRP is empty after dropna"

    def test_vrp_formula(self, returns_500, vix_df_500):
        """Spot-check: VRP = IV² − RV; IV² = (VIX/100)² × 252."""
        vix = vix_df_500["vix"].reindex(returns_500.index).ffill()
        rv  = realized_variance_cc(returns_500, window=20)
        iv2 = (vix / 100.0) ** 2 * 252

        expected = (iv2 - rv).dropna()
        actual   = variance_risk_premium(returns_500, vix).dropna()

        common = expected.index.intersection(actual.index)
        pd.testing.assert_series_equal(
            actual.loc[common].rename("vrp"),
            expected.loc[common].rename("vrp"),
            check_names=False,
            atol=1e-12,
        )

    def test_vrp_with_lag(self, returns_500, vix_df_500):
        """lag > 0 should shift RV back, changing VRP values."""
        vix  = vix_df_500["vix"]
        vrp0 = variance_risk_premium(returns_500, vix, lag=0).dropna()
        vrp5 = variance_risk_premium(returns_500, vix, lag=5).dropna()
        assert not vrp0.equals(vrp5), "lag=5 should produce different VRP than lag=0"


# ---------------------------------------------------------------------------
# FR-05: VIX term structure
# ---------------------------------------------------------------------------

class TestVIXTermStructure:

    def test_columns_present(self, vix_df_500):
        ts = vix_term_structure_features(vix_df_500)
        assert "spot_ratio" in ts.columns
        assert "term_ratio" in ts.columns

    def test_ratios_positive(self, vix_df_500):
        ts = vix_term_structure_features(vix_df_500).dropna()
        assert (ts["spot_ratio"] > 0).all()
        assert (ts["term_ratio"] > 0).all()

    def test_ratios_clipped(self, vix_df_500):
        ts = vix_term_structure_features(vix_df_500).dropna()
        assert (ts["spot_ratio"] >= 0.3).all() and (ts["spot_ratio"] <= 3.0).all()
        assert (ts["term_ratio"] >= 0.3).all() and (ts["term_ratio"] <= 3.0).all()

    def test_missing_vix9d_returns_nan_spot_ratio(self):
        """If VIX9D is missing the spot_ratio column should be NaN."""
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        df  = pd.DataFrame({"vix": np.full(50, 20.0), "vix3m": np.full(50, 18.0)}, index=idx)
        ts  = vix_term_structure_features(df)
        assert ts["spot_ratio"].isna().all(), "spot_ratio should be NaN when VIX9D is absent"

    def test_backwardation_condition(self):
        """term_ratio = VIX/VIX3M > 1 signals backwardation (high-vol)."""
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        df  = pd.DataFrame({
            "vix":   np.full(10, 30.0),   # elevated
            "vix9d": np.full(10, 28.0),
            "vix3m": np.full(10, 20.0),   # term_ratio = 30/20 = 1.5
        }, index=idx)
        ts = vix_term_structure_features(df)
        assert (ts["term_ratio"] > 1.0).all(), "term_ratio should be > 1 in backwardation"


# ---------------------------------------------------------------------------
# FR-07: Downside risk & skewness
# ---------------------------------------------------------------------------

class TestDownsideSkewFeatures:

    def test_columns_present(self, returns_500):
        ds = downside_skew_features(returns_500)
        assert "downside_vol" in ds.columns
        assert "skewness_20d" in ds.columns

    def test_downside_vol_positive(self, returns_500):
        ds = downside_skew_features(returns_500).dropna()
        assert (ds["downside_vol"] >= 0).all(), "downside_vol must be non-negative"

    def test_downside_vol_annualised(self, returns_500):
        """downside_vol should be in a plausible annualised range."""
        ds = downside_skew_features(returns_500).dropna()
        mean_dv = ds["downside_vol"].mean()
        assert 0.02 < mean_dv < 1.0, f"downside_vol mean {mean_dv:.3f} looks implausible"

    def test_skewness_range(self, returns_500):
        """20-day skewness should be in (-5, 5) for normal-ish data."""
        ds = downside_skew_features(returns_500).dropna()
        assert ds["skewness_20d"].abs().max() < 5.0, "skewness_20d has implausibly large value"

    def test_custom_window(self, returns_500):
        """Window parameter should be respected."""
        ds10 = downside_skew_features(returns_500, window=10).dropna()
        ds60 = downside_skew_features(returns_500, window=60).dropna()
        # 60-day window produces smoother (lower std) downside_vol
        assert ds60["downside_vol"].std() < ds10["downside_vol"].std()


# ---------------------------------------------------------------------------
# Parkinson estimator
# ---------------------------------------------------------------------------

class TestParkinsonEstimator:

    def test_returns_series(self, prices_500):
        rv = realized_variance_parkinson(prices_500)
        assert isinstance(rv, pd.Series)

    def test_positive_values(self, prices_500):
        rv = realized_variance_parkinson(prices_500).dropna()
        assert (rv > 0).all(), "Parkinson RV must be positive"

    def test_name(self, prices_500):
        rv = realized_variance_parkinson(prices_500)
        assert rv.name == "rv_parkinson"

    def test_missing_high_low_returns_nan(self):
        """Without high/low columns the estimator should return all-NaN."""
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        df  = pd.DataFrame({"close": np.ones(30)}, index=idx)
        rv  = realized_variance_parkinson(df)
        assert rv.isna().all(), "Expected all-NaN when high/low are absent"


# ---------------------------------------------------------------------------
# Realised variance close-to-close
# ---------------------------------------------------------------------------

class TestRealisedVarianceCC:

    def test_non_negative(self, returns_500):
        rv = realized_variance_cc(returns_500, window=20).dropna()
        assert (rv >= 0).all()

    def test_window_respected(self, returns_500):
        """With window=20, first 19 values must be NaN."""
        rv = realized_variance_cc(returns_500, window=20)
        assert rv.iloc[:19].isna().all()
        assert rv.iloc[19:].notna().any()

    def test_annualisation(self, returns_500):
        rv = realized_variance_cc(returns_500, window=20).dropna()
        # σ=0.01 daily → RV ≈ 252 × 0.01² = 0.0252
        mean_rv = rv.mean()
        assert 0.005 < mean_rv < 0.15, f"RV mean {mean_rv:.5f} looks implausible"


# ---------------------------------------------------------------------------
# build_features — tier system  (FR-04 to FR-07 integration)
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_tier1_columns(self, returns_500):
        """Tier 1 must include rv_short/medium/long + vol_ratio."""
        features = build_features(returns_500)
        for col in ["rv_short", "rv_medium", "rv_long", "vol_ratio"]:
            assert col in features.columns, f"Tier-1 col missing: {col}"

    def test_tier2_columns(self, returns_500, vix_df_500):
        """Tier 2 adds vrp, spot_ratio, term_ratio, downside_vol."""
        features = build_features(returns_500, vix_df=vix_df_500)
        for col in ["vrp", "spot_ratio", "term_ratio", "downside_vol"]:
            assert col in features.columns, f"Tier-2 col missing: {col}"

    def test_tier3_columns(self, returns_500, vix_df_500, prices_500):
        """Tier 3 adds rv_parkinson + skewness_20d."""
        features = build_features(returns_500, prices=prices_500, vix_df=vix_df_500)
        for col in ["rv_parkinson", "skewness_20d"]:
            assert col in features.columns, f"Tier-3 col missing: {col}"

    def test_no_nan_in_output(self, returns_500, vix_df_500):
        features = build_features(returns_500, vix_df=vix_df_500)
        assert not features.isnull().any().any(), "build_features output contains NaN"

    def test_no_inf_in_output(self, returns_500, vix_df_500):
        features = build_features(returns_500, vix_df=vix_df_500)
        assert not np.isinf(features.values).any(), "build_features output contains Inf"

    def test_returns_column_present(self, returns_500):
        features = build_features(returns_500)
        assert "returns" in features.columns

    def test_index_is_datetimeindex(self, returns_500):
        features = build_features(returns_500)
        assert isinstance(features.index, pd.DatetimeIndex)

    def test_shape_shrinks_from_warmup(self, returns_500):
        """build_features must drop warm-up rows (60d max)."""
        features = build_features(returns_500)
        assert len(features) < len(returns_500)


class TestGetFeatureCols:

    def test_tier1_subset(self, returns_500):
        features = build_features(returns_500)
        cols = get_feature_cols(features, tier=1)
        assert set(cols) == {"rv_short", "rv_medium", "rv_long", "vol_ratio"}

    def test_tier2_superset_of_tier1(self, returns_500, vix_df_500):
        features = build_features(returns_500, vix_df=vix_df_500)
        t1 = set(get_feature_cols(features, tier=1))
        t2 = set(get_feature_cols(features, tier=2))
        assert t1.issubset(t2), "Tier-2 should be a superset of Tier-1"

    def test_missing_cols_excluded(self, returns_500):
        """If VIX data is absent, tier=2 request should silently drop VIX cols."""
        features = build_features(returns_500)   # no vix_df
        cols = get_feature_cols(features, tier=2)
        for c in ["vrp", "spot_ratio", "term_ratio"]:
            assert c not in cols, f"VIX col {c} should not appear without vix_df"

    def test_cols_are_subset_of_df_columns(self, returns_500, vix_df_500):
        features = build_features(returns_500, vix_df=vix_df_500)
        for tier in [1, 2, 3]:
            cols = get_feature_cols(features, tier=tier)
            for c in cols:
                assert c in features.columns, f"Column {c} in tier={tier} not in DataFrame"
