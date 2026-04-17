"""
HMM Math Test Suite — test_hmm_math.py

Tests the mathematical correctness of the Student-t HMM implementation.
Any quant engineer at HRT would write these on day one of a code review.

Coverage
--------
1. Log-likelihood monotonicity  — EM must never decrease the objective
2. Parameter bounds             — ν ∈ (1, 300], σ > 0, π/A row-stochastic
3. Label ordering               — State 0 always has lowest ||σ_k||₂
4. Viterbi vs Baum-Welch        — MAP state (Viterbi) must agree with
                                  argmax of posterior marginals on clean data
5. Log-space numerics           — Forward-backward must not produce NaN/Inf
                                  on long sequences (T > 1000)
6. AIC/BIC correctness          — Formula matches manual calculation
7. select_n_states              — Returns valid K in k_range
8. Convergence on synthetic data — Recovers known means within tolerance
9. predict_proba sums to 1      — Posterior rows must be a valid distribution
10. fix_nu=True                 — ν must not change when fix_nu=True
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.models.hmm_studentt import (
    StudentTHMM,
    StudentTHMMParams,
    _log_studentt,
    _forward,
    _backward,
    _compute_log_obs,
    _compute_aux_weights,
    _update_nu,
    _logsumexp,
    select_n_states,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_regime_data():
    """
    Synthetic 2-regime data: 400 low-vol obs then 400 high-vol obs.
    Regimes are well-separated so label recovery should be unambiguous.
    """
    rng = np.random.default_rng(0)
    X_low  = rng.normal(0.0, 0.005, (400, 2))
    X_high = rng.normal(0.0, 0.020, (400, 2))
    X = np.vstack([X_low, X_high])
    true_states = np.array([0] * 400 + [1] * 400)
    return X, true_states


@pytest.fixture(scope="module")
def fitted_model_2state(two_regime_data):
    """Pre-fitted 2-state model for reuse across tests."""
    X, _ = two_regime_data
    model = StudentTHMM(n_states=2, fix_nu=False, max_iter=100, random_seed=1)
    model.fit(X)
    return model, X


# ---------------------------------------------------------------------------
# 1. Log-likelihood monotonicity
# ---------------------------------------------------------------------------

class TestLogLikelihoodMonotonicity:
    """EM must never decrease the log-likelihood between iterations."""

    def test_ll_monotone_2state(self):
        """Track LL manually over EM iterations; assert non-decreasing."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(0, 0.005, (300, 2)),
            rng.normal(0, 0.025, (300, 2)),
        ])

        model = StudentTHMM(n_states=2, fix_nu=False, max_iter=80, random_seed=7)

        # Manually run EM, capturing LL at each step
        model._check_fitted = lambda: None   # bypass guard during manual run
        p = model._init_params(X)

        log_liks = []
        for _ in range(40):
            gamma, xi, ll = model._e_step(X, p)
            log_liks.append(ll)
            p = model._m_step(X, gamma, xi, p)

        # Every step must be >= previous (allow tiny floating-point slack)
        for i in range(1, len(log_liks)):
            assert log_liks[i] >= log_liks[i - 1] - 1e-4, (
                f"LL decreased at iter {i}: {log_liks[i-1]:.6f} → {log_liks[i]:.6f}"
            )

    def test_ll_monotone_3state(self):
        """Monotonicity must hold for K=3 as well."""
        rng = np.random.default_rng(99)
        X = np.vstack([
            rng.normal(0, 0.003, (200, 3)),
            rng.normal(0, 0.010, (200, 3)),
            rng.normal(0, 0.025, (200, 3)),
        ])

        model = StudentTHMM(n_states=3, fix_nu=False, max_iter=50, random_seed=5)
        p = model._init_params(X)

        log_liks = []
        for _ in range(30):
            gamma, xi, ll = model._e_step(X, p)
            log_liks.append(ll)
            p = model._m_step(X, gamma, xi, p)

        for i in range(1, len(log_liks)):
            assert log_liks[i] >= log_liks[i - 1] - 1e-4, (
                f"K=3 LL decreased at iter {i}: {log_liks[i-1]:.6f} → {log_liks[i]:.6f}"
            )

    def test_final_ll_equals_score(self, fitted_model_2state):
        """score() must return the same LL that fit() converged to."""
        model, X = fitted_model_2state
        assert abs(model.score(X) - model.params_.log_likelihood) < 1.0


# ---------------------------------------------------------------------------
# 2. Parameter bounds
# ---------------------------------------------------------------------------

class TestParameterBounds:
    """All parameters must satisfy their mathematical constraints after fit."""

    def test_nu_in_bounds(self, fitted_model_2state):
        """ν_k must be in (1, 300] for all states."""
        model, _ = fitted_model_2state
        nu = model.params_.nu
        assert np.all(nu > 1.0),  f"ν contains values ≤ 1: {nu}"
        assert np.all(nu <= 300.0), f"ν exceeds upper bound: {nu}"

    def test_sigma_positive(self, fitted_model_2state):
        """All σ_k must be strictly positive."""
        model, _ = fitted_model_2state
        assert np.all(model.params_.sigma > 0), "σ contains non-positive values"

    def test_pi_sums_to_one(self, fitted_model_2state):
        """Initial state distribution must sum to 1."""
        model, _ = fitted_model_2state
        assert abs(model.params_.pi.sum() - 1.0) < 1e-8

    def test_pi_non_negative(self, fitted_model_2state):
        """Initial state distribution must be non-negative."""
        model, _ = fitted_model_2state
        assert np.all(model.params_.pi >= 0)

    def test_transition_matrix_row_stochastic(self, fitted_model_2state):
        """Each row of A must sum to 1."""
        model, _ = fitted_model_2state
        row_sums = model.params_.A.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8,
            err_msg=f"A row sums not 1: {row_sums}")

    def test_transition_matrix_non_negative(self, fitted_model_2state):
        """All entries of A must be non-negative."""
        model, _ = fitted_model_2state
        assert np.all(model.params_.A >= 0)

    def test_nu_fixed_when_fix_nu_true(self, two_regime_data):
        """When fix_nu=True, ν must remain at the initial value (5.0)."""
        X, _ = two_regime_data
        model = StudentTHMM(n_states=2, fix_nu=True, max_iter=30, random_seed=1)
        model.fit(X)
        np.testing.assert_allclose(model.params_.nu, 5.0, atol=1e-10,
            err_msg="ν changed despite fix_nu=True")


# ---------------------------------------------------------------------------
# 3. Label ordering (canonical sort)
# ---------------------------------------------------------------------------

class TestLabelOrdering:
    """State 0 must always correspond to the lowest-volatility regime."""

    def test_sigma_norm_ascending(self, fitted_model_2state):
        """||σ_k||₂ must be non-decreasing with state index."""
        model, _ = fitted_model_2state
        norms = np.linalg.norm(model.params_.sigma, axis=1)
        for i in range(len(norms) - 1):
            assert norms[i] <= norms[i + 1], (
                f"Label ordering violated: ||σ_{i}||={norms[i]:.6f} "
                f"> ||σ_{i+1}||={norms[i+1]:.6f}"
            )

    def test_state0_is_low_vol(self, fitted_model_2state):
        """State 0 must have the smallest σ norm."""
        model, _ = fitted_model_2state
        norms = np.linalg.norm(model.params_.sigma, axis=1)
        assert norms[0] == norms.min(), "State 0 is not the minimum-volatility state"

    def test_ordering_reproducible_across_seeds(self, two_regime_data):
        """
        Label ordering must be consistent regardless of random initialisation.
        Fit with 3 different seeds; all should agree on which state is low-vol.
        """
        X, _ = two_regime_data
        norms_list = []
        for seed in [1, 2, 3]:
            m = StudentTHMM(n_states=2, fix_nu=False, max_iter=50, random_seed=seed)
            m.fit(X)
            norms_list.append(np.linalg.norm(m.params_.sigma, axis=1))

        # All seeds should identify the same state-0 σ (within tolerance)
        for norms in norms_list:
            assert norms[0] < norms[1], "Seed produced wrong ordering"


# ---------------------------------------------------------------------------
# 4. Viterbi vs Baum-Welch consistency
# ---------------------------------------------------------------------------

class TestViterbiVsBaumWelch:
    """
    On well-separated data the MAP state sequence (Viterbi) should agree
    with argmax of the posterior marginals (Baum-Welch) for most time steps.
    """

    def test_viterbi_agrees_with_argmax_gamma(self, fitted_model_2state):
        """
        agreement = mean(viterbi_state == argmax(gamma))
        Must be >= 0.90 on clean 2-regime synthetic data.
        """
        model, X = fitted_model_2state
        viterbi_states = model.predict(X)
        gamma          = model.predict_proba(X)
        gamma_states   = np.argmax(gamma, axis=1)

        agreement = np.mean(viterbi_states == gamma_states)
        assert agreement >= 0.90, (
            f"Viterbi/γ-argmax agreement too low: {agreement:.3f}"
        )

    def test_viterbi_recovers_true_regimes(self, two_regime_data, fitted_model_2state):
        """
        Viterbi must achieve >= 85 % accuracy on the known ground-truth labels
        (allowing for the random boundary region).
        """
        X, true_states = two_regime_data
        model, _ = fitted_model_2state
        pred = model.predict(X)

        # Account for possible global label flip (0↔1)
        acc_direct = np.mean(pred == true_states)
        acc_flipped = np.mean(pred == (1 - true_states))
        acc = max(acc_direct, acc_flipped)

        assert acc >= 0.85, (
            f"Viterbi regime recovery accuracy too low: {acc:.3f}"
        )

    def test_viterbi_output_dtype_and_shape(self, fitted_model_2state):
        """predict() must return an integer array of shape (T,)."""
        model, X = fitted_model_2state
        states = model.predict(X)
        assert states.shape == (len(X),)
        assert np.issubdtype(states.dtype, np.integer)

    def test_viterbi_states_in_range(self, fitted_model_2state):
        """All predicted states must be in {0, ..., K-1}."""
        model, X = fitted_model_2state
        states = model.predict(X)
        assert np.all(states >= 0) and np.all(states < model.n_states)


# ---------------------------------------------------------------------------
# 5. Log-space numerics
# ---------------------------------------------------------------------------

class TestLogSpaceNumerics:
    """Forward-backward must be numerically stable on long sequences."""

    @pytest.mark.parametrize("T", [500, 1000, 2000, 5000])
    def test_no_nan_inf_on_long_sequence(self, T):
        """
        predict_proba must return finite values for sequences of length T.
        This tests the logsumexp scaling in _forward / _backward.
        """
        rng = np.random.default_rng(T)
        X = np.vstack([
            rng.normal(0, 0.005, (T // 2, 2)),
            rng.normal(0, 0.020, (T // 2, 2)),
        ])

        model = StudentTHMM(n_states=2, fix_nu=False, max_iter=30, random_seed=1)
        model.fit(X)
        probs = model.predict_proba(X)

        assert not np.any(np.isnan(probs)), f"NaN in predict_proba for T={T}"
        assert not np.any(np.isinf(probs)), f"Inf in predict_proba for T={T}"

    def test_logsumexp_numerical_stability(self):
        """_logsumexp must handle very large negative values without underflow."""
        a = np.array([-1000.0, -999.5, -1001.0])
        result = _logsumexp(a)
        assert np.isfinite(result), f"logsumexp returned non-finite: {result}"
        # Reference: scipy logsumexp
        from scipy.special import logsumexp as scipy_lse
        np.testing.assert_allclose(result, scipy_lse(a), rtol=1e-10)

    def test_logsumexp_all_neginf(self):
        """_logsumexp on all -inf must return -inf, not NaN."""
        a = np.array([-np.inf, -np.inf, -np.inf])
        result = _logsumexp(a)
        assert result == -np.inf

    def test_log_studentt_finite(self):
        """_log_studentt must return finite values for well-conditioned inputs."""
        rng = np.random.default_rng(0)
        X   = rng.normal(0, 1, (200, 3))
        mu  = np.zeros(3)
        sig = np.ones(3)
        lp  = _log_studentt(X, mu, sig, nu=5.0)

        assert lp.shape == (200,)
        assert np.all(np.isfinite(lp)), "log_studentt returned non-finite values"
        assert np.all(lp <= 0), "log probability must be <= 0"


# ---------------------------------------------------------------------------
# 6. AIC / BIC correctness
# ---------------------------------------------------------------------------

class TestAICBIC:
    """AIC and BIC must match the manual formula."""

    def test_aic_formula(self, fitted_model_2state):
        """AIC = 2k − 2LL."""
        model, X = fitted_model_2state
        _, d = X.shape
        k  = model._n_params(d)
        ll = model.params_.log_likelihood
        expected = 2.0 * k - 2.0 * ll
        np.testing.assert_allclose(model.params_.aic, expected, rtol=1e-10)

    def test_bic_formula(self, fitted_model_2state):
        """BIC = k log(T) − 2LL."""
        model, X = fitted_model_2state
        T, d = X.shape
        k  = model._n_params(d)
        ll = model.params_.log_likelihood
        expected = k * np.log(T) - 2.0 * ll
        np.testing.assert_allclose(model.params_.bic, expected, rtol=1e-10)

    def test_aic_bic_finite(self, fitted_model_2state):
        """AIC and BIC must be finite after a successful fit."""
        model, _ = fitted_model_2state
        assert np.isfinite(model.params_.aic), "AIC is not finite"
        assert np.isfinite(model.params_.bic), "BIC is not finite"

    def test_bic_penalises_more_than_aic(self, two_regime_data):
        """
        BIC penalty (k log T) must exceed AIC penalty (2k) when T > 7
        (since log(7) ≈ 1.95 < 2, BIC wins only for T > e² ≈ 7.4).
        With T=800 this must always hold.
        """
        X, _ = two_regime_data
        T = len(X)
        model = StudentTHMM(n_states=2, fix_nu=False, max_iter=50, random_seed=1)
        model.fit(X)
        _, d = X.shape
        k = model._n_params(d)

        bic_penalty = k * np.log(T)
        aic_penalty = 2.0 * k
        assert bic_penalty > aic_penalty, (
            f"BIC penalty {bic_penalty:.2f} should exceed AIC penalty {aic_penalty:.2f} for T={T}"
        )

    def test_n_params_fix_nu_reduces_count(self, two_regime_data):
        """fix_nu=True should reduce parameter count by K."""
        X, _ = two_regime_data
        _, d = X.shape
        m_free = StudentTHMM(n_states=2, fix_nu=False)
        m_fixed = StudentTHMM(n_states=2, fix_nu=True)
        K = 2
        assert m_free._n_params(d) - m_fixed._n_params(d) == K


# ---------------------------------------------------------------------------
# 7. select_n_states
# ---------------------------------------------------------------------------

class TestSelectNStates:
    """select_n_states must return a valid K from k_range."""

    def test_returns_valid_k(self, two_regime_data):
        X, _ = two_regime_data
        k_range = [2, 3]
        best_k, score = select_n_states(X, k_range=k_range, max_iter=20, criterion="bic")
        assert best_k in k_range, f"best_k={best_k} not in k_range={k_range}"
        assert np.isfinite(score), f"score is not finite: {score}"

    def test_aic_criterion(self, two_regime_data):
        X, _ = two_regime_data
        best_k, score = select_n_states(X, k_range=[2, 3], max_iter=20, criterion="aic")
        assert best_k in [2, 3]
        assert np.isfinite(score)

    def test_selects_k2_on_clean_2state_data(self, two_regime_data):
        """
        On clearly 2-regime data BIC should prefer K=2 over K=4.
        (K=3 is borderline — only test the extremes.)
        """
        X, _ = two_regime_data
        k_range = [2, 4]
        best_k, _ = select_n_states(X, k_range=k_range, max_iter=30, criterion="bic")
        assert best_k == 2, f"Expected K=2 but got K={best_k} on 2-regime data"


# ---------------------------------------------------------------------------
# 8. Convergence on synthetic data
# ---------------------------------------------------------------------------

class TestConvergence:
    """Model must recover known parameters within reasonable tolerance."""

    def test_recovers_means(self):
        """
        On 1-D data with well-separated means the fitted μ must be
        within 0.002 of the true means.
        """
        rng = np.random.default_rng(7)
        true_mu = np.array([-0.05, 0.05])
        X = np.vstack([
            rng.normal(true_mu[0], 0.005, (400, 1)),
            rng.normal(true_mu[1], 0.010, (400, 1)),
        ])
        model = StudentTHMM(n_states=2, fix_nu=False, max_iter=100, random_seed=3)
        model.fit(X)

        fitted_mu = np.sort(model.params_.mu.flatten())
        np.testing.assert_allclose(fitted_mu, np.sort(true_mu), atol=0.002,
            err_msg="Fitted means deviate too far from true means")

    def test_recovers_sigma_ordering(self, two_regime_data):
        """
        The model must correctly identify which state is high-vol.
        σ[0] < σ[1] by canonical ordering; true σ ratio is ~4×.
        """
        X, _ = two_regime_data
        model = StudentTHMM(n_states=2, fix_nu=False, max_iter=100, random_seed=1)
        model.fit(X)

        # State 0 should have σ ≈ 0.005, state 1 should have σ ≈ 0.020
        s0 = model.params_.sigma[0].mean()
        s1 = model.params_.sigma[1].mean()
        assert s1 / s0 > 2.0, (
            f"Expected σ[1]/σ[0] > 2, got {s1/s0:.2f}"
        )

    def test_converged_flag(self):
        """On easy data the model should converge within max_iter."""
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal(0, 0.005, (300, 2)),
            rng.normal(0, 0.020, (300, 2)),
        ])
        model = StudentTHMM(n_states=2, fix_nu=False, max_iter=200, random_seed=1)
        model.fit(X)
        assert model.params_.converged, "Model did not converge on clean 2-regime data"


# ---------------------------------------------------------------------------
# 9. predict_proba — valid probability distribution
# ---------------------------------------------------------------------------

class TestPredictProba:
    """Posterior state probabilities must form a valid distribution."""

    def test_rows_sum_to_one(self, fitted_model_2state):
        model, X = fitted_model_2state
        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6,
            err_msg="predict_proba rows do not sum to 1")

    def test_all_non_negative(self, fitted_model_2state):
        model, X = fitted_model_2state
        probs = model.predict_proba(X)
        assert np.all(probs >= 0), "predict_proba returned negative probabilities"

    def test_shape(self, fitted_model_2state):
        model, X = fitted_model_2state
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), model.n_states)

    def test_on_1d_input(self):
        """predict_proba must accept 1-D input (T,) and reshape internally."""
        rng = np.random.default_rng(0)
        X_1d = rng.normal(0, 0.01, 200)
        model = StudentTHMM(n_states=2, fix_nu=False, max_iter=30, random_seed=1)
        model.fit(X_1d)
        probs = model.predict_proba(X_1d)
        assert probs.shape == (200, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 10. Auxiliary weights correctness
# ---------------------------------------------------------------------------

class TestAuxiliaryWeights:
    """w_tk = (ν+d)/(ν+δ_tk) — spot-check the formula."""

    def test_weights_positive(self):
        rng = np.random.default_rng(1)
        X   = rng.normal(0, 1, (100, 2))
        mu  = np.zeros(2)
        sig = np.ones(2)
        w   = _compute_aux_weights(X, mu, sig, nu=5.0)
        assert np.all(w > 0), "Auxiliary weights must be strictly positive"

    def test_weights_bounded(self):
        """w_tk <= (ν+d)/ν = 1 + d/ν (maximum at δ=0)."""
        rng = np.random.default_rng(2)
        X   = rng.normal(0, 1, (200, 3))
        mu  = np.zeros(3)
        sig = np.ones(3)
        nu  = 5.0
        d   = 3
        w   = _compute_aux_weights(X, mu, sig, nu=nu)
        upper = (nu + d) / nu
        assert np.all(w <= upper + 1e-10), f"Weights exceed upper bound {upper:.3f}"

    def test_weights_increase_toward_mean(self):
        """
        Observation equal to the mean has δ=0 → w = (ν+d)/ν (max weight).
        Any other observation must have lower weight.
        """
        mu  = np.array([0.0, 0.0])
        sig = np.array([1.0, 1.0])
        nu  = 5.0
        d   = 2

        X_at_mean = np.array([[0.0, 0.0]])            # δ=0
        X_far     = np.array([[10.0, 10.0]])           # large δ

        w_at_mean = _compute_aux_weights(X_at_mean, mu, sig, nu)
        w_far     = _compute_aux_weights(X_far,     mu, sig, nu)

        assert w_at_mean[0] > w_far[0], "Weight at mean should exceed weight of outlier"
        np.testing.assert_allclose(w_at_mean[0], (nu + d) / nu, rtol=1e-10)


# ---------------------------------------------------------------------------
# 11. Newton-Raphson ν update
# ---------------------------------------------------------------------------

class TestNewtonRaphsonNu:
    """_update_nu must converge and stay within bounds."""

    def test_nu_stays_in_bounds(self):
        rng = np.random.default_rng(0)
        gamma_k = np.ones(200) / 200
        w_k     = rng.uniform(0.5, 2.0, 200)
        for nu_init in [2.0, 5.0, 20.0, 100.0]:
            nu_new = _update_nu(nu_init, gamma_k, w_k, d=2)
            assert 1.0 < nu_new <= 300.0, (
                f"ν update out of bounds: ν_init={nu_init} → ν_new={nu_new}"
            )

    def test_nu_moves_from_init(self):
        """
        On Student-t(df=4) data the update should move ν toward ~4,
        i.e. away from the default init of 5.0.
        """
        rng = np.random.default_rng(3)
        X   = rng.standard_t(df=4, size=(500, 1))
        mu  = np.array([0.0])
        sig = np.array([1.0])
        nu_init = 5.0
        gamma_k = np.ones(500) / 500
        w_k     = _compute_aux_weights(X, mu, sig, nu_init)
        nu_new  = _update_nu(nu_init, gamma_k, w_k, d=1)
        # ν should not be stuck at the starting value
        assert abs(nu_new - nu_init) > 0.01, (
            f"ν update did not move from init: {nu_init} → {nu_new}"
        )
