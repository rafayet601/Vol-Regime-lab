"""
Student-t HMM with full Baum-Welch EM — implemented from scratch.

Mathematical references:
  - Dempster, Laird & Rubin (1977). EM Algorithm. JRSS-B.
  - Liu & Rubin (1994). ECME algorithm. Biometrika.
  - Hamilton (1989). Business Cycle. Econometrica.

No wrapper libraries (hmmlearn, pomegranate) are used.
All forward-backward numerics are in log-space via logsumexp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.special import digamma, gammaln, polygamma

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class StudentTHMMParams:
    """Fitted parameters for a K-state diagonal Student-t HMM."""
    pi: np.ndarray          # (K,)  initial state distribution
    A: np.ndarray           # (K, K) row-stochastic transition matrix
    mu: np.ndarray          # (K, d) emission means
    sigma: np.ndarray       # (K, d) emission std devs  (diagonal cov)
    nu: np.ndarray          # (K,)  degrees of freedom per state
    log_likelihood: float = -np.inf
    n_iter: int = 0
    converged: bool = False
    aic: float = field(default=np.inf, repr=False)
    bic: float = field(default=np.inf, repr=False)


# ---------------------------------------------------------------------------
# Log-space utilities
# ---------------------------------------------------------------------------

def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    a_max = np.max(a)
    if np.isneginf(a_max):
        return -np.inf
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def _log_studentt(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: float) -> np.ndarray:
    """
    Log-density of a diagonal multivariate Student-t for each observation.

    Parameters
    ----------
    X     : (T, d)
    mu    : (d,)
    sigma : (d,)  — scale (std dev), not variance
    nu    : float — degrees of freedom

    Returns
    -------
    log_p : (T,)
    """
    T, d = X.shape
    sigma = np.maximum(sigma, 1e-8)

    # Mahalanobis² (diagonal): sum over dims of ((x - mu)/sigma)²
    z = (X - mu) / sigma          # (T, d)
    delta = np.sum(z ** 2, axis=1)  # (T,)

    # Log-normalising constant for multivariate t with diagonal Σ
    #   = sum_j [ log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) - log σ_j ]
    log_norm = d * (
        gammaln(0.5 * (nu + 1))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
    ) - np.sum(np.log(sigma))

    log_p = log_norm - 0.5 * (nu + d) * np.log1p(delta / nu)
    return log_p


# ---------------------------------------------------------------------------
# Forward-backward (log-space)
# ---------------------------------------------------------------------------

def _forward(log_pi: np.ndarray, log_A: np.ndarray, log_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Log-space forward algorithm.

    Returns
    -------
    alpha : (T, K)  log forward variables
    log_c : (T,)    log scaling constants  (log P(x_1..t))
    """
    T, K = log_obs.shape
    alpha = np.full((T, K), -np.inf)
    log_c = np.zeros(T)

    alpha[0] = log_pi + log_obs[0]
    log_c[0] = _logsumexp(alpha[0])
    alpha[0] -= log_c[0]

    for t in range(1, T):
        for k in range(K):
            alpha[t, k] = _logsumexp(alpha[t - 1] + log_A[:, k]) + log_obs[t, k]
        log_c[t] = _logsumexp(alpha[t])
        alpha[t] -= log_c[t]

    return alpha, log_c


def _backward(log_A: np.ndarray, log_obs: np.ndarray, log_c: np.ndarray) -> np.ndarray:
    """
    Log-space backward algorithm.

    Returns
    -------
    beta : (T, K)  log backward variables (scaled by same log_c)
    """
    T, K = log_obs.shape
    beta = np.full((T, K), -np.inf)
    beta[T - 1] = 0.0  # log(1)

    for t in range(T - 2, -1, -1):
        for i in range(K):
            beta[t, i] = _logsumexp(log_A[i] + log_obs[t + 1] + beta[t + 1])
        beta[t] -= log_c[t + 1]

    return beta


def _compute_log_obs(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    Compute log emission probabilities for all states.

    Returns
    -------
    log_obs : (T, K)
    """
    K = mu.shape[0]
    T = X.shape[0]
    log_obs = np.zeros((T, K))
    for k in range(K):
        log_obs[:, k] = _log_studentt(X, mu[k], sigma[k], nu[k])
    return log_obs


# ---------------------------------------------------------------------------
# Auxiliary weights (E-step for Student-t)
# ---------------------------------------------------------------------------

def _compute_aux_weights(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: float) -> np.ndarray:
    """
    w_tk = (ν + d) / (ν + δ_tk)  where δ_tk = Mahalanobis²

    Returns
    -------
    w : (T,)
    """
    d = X.shape[1]
    sigma = np.maximum(sigma, 1e-8)
    z = (X - mu) / sigma
    delta = np.sum(z ** 2, axis=1)
    return (nu + d) / (nu + delta)


# ---------------------------------------------------------------------------
# Newton-Raphson M-step for ν
# ---------------------------------------------------------------------------

def _update_nu(nu_init: float, gamma_k: np.ndarray, w_k: np.ndarray,
               d: int, max_iter: int = 50, tol: float = 1e-6) -> float:
    """
    Solve for ν_k via Newton-Raphson on:

        g(ν) = log(ν/2) − ψ(ν/2) + 1 + (1/N_k) Σ_t γ_tk [log w_tk − w_tk] = 0

    where ψ = digamma, and the last term is the E-step sufficient statistic.
    """
    Nk = np.sum(gamma_k)
    if Nk < 1e-10:
        return nu_init

    # Sufficient statistic: (1/Nk) Σ γ_tk [log w_tk − w_tk]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_w = np.where(w_k > 0, np.log(w_k), -np.inf)
    S = np.sum(gamma_k * (log_w - w_k)) / Nk

    nu = float(nu_init)
    for _ in range(max_iter):
        g = np.log(nu / 2.0) - digamma(nu / 2.0) + 1.0 + S
        gp = 1.0 / nu - 0.5 * polygamma(1, nu / 2.0)   # trigamma
        if abs(gp) < 1e-15:
            break
        step = g / gp
        nu_new = nu - step
        # Keep ν in (1, 300) — outside this range Student-t ≈ Gaussian
        nu_new = float(np.clip(nu_new, 1.01, 300.0))
        if abs(nu_new - nu) < tol:
            nu = nu_new
            break
        nu = nu_new

    return nu


# ---------------------------------------------------------------------------
# Main HMM class
# ---------------------------------------------------------------------------

class StudentTHMM:
    """
    K-state Hidden Markov Model with diagonal Student-t emissions.

    All HMM math (forward-backward, Viterbi, EM) is implemented from
    scratch in log-space.  No hmmlearn or pomegranate dependency.

    Parameters
    ----------
    n_states : int
        Number of hidden states K.
    fix_nu : bool
        If True, degrees-of-freedom are not updated in the M-step
        (equivalent to Gaussian HMM when ν is large).  Default False.
    max_iter : int
        Maximum Baum-Welch iterations.
    tol : float
        Convergence threshold on log-likelihood change.
    random_seed : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 2,
        fix_nu: bool = False,
        max_iter: int = 200,
        tol: float = 1e-6,
        random_seed: Optional[int] = 42,
    ):
        self.n_states = n_states
        self.fix_nu = fix_nu
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed

        self.params_: Optional[StudentTHMMParams] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self, X: np.ndarray) -> StudentTHMMParams:
        """K-means (10 iter) init for μ; state-conditional σ; uniform A, π."""
        rng = np.random.default_rng(self.random_seed)
        T, d = X.shape
        K = self.n_states

        # --- K-means (10 iterations, random init) ---
        idx = rng.choice(T, K, replace=False)
        mu = X[idx].copy()  # (K, d)

        for _ in range(10):
            dists = np.array([np.sum(((X - mu[k]) ** 2), axis=1) for k in range(K)])  # (K, T)
            labels = np.argmin(dists, axis=0)  # (T,)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 1:
                    mu[k] = X[mask].mean(axis=0)

        # State-conditional std
        dists = np.array([np.sum(((X - mu[k]) ** 2), axis=1) for k in range(K)])
        labels = np.argmin(dists, axis=0)
        sigma = np.ones((K, d))
        for k in range(K):
            mask = labels == k
            if mask.sum() > 1:
                sigma[k] = np.std(X[mask], axis=0)
        sigma = np.maximum(sigma, 1e-4)

        # Uniform transition matrix + initial probs
        A = np.full((K, K), 0.1 / (K - 1)) if K > 1 else np.ones((1, 1))
        np.fill_diagonal(A, 0.9)
        A /= A.sum(axis=1, keepdims=True)

        pi = np.ones(K) / K
        nu = np.full(K, 5.0)

        return StudentTHMMParams(pi=pi, A=A, mu=mu, sigma=sigma, nu=nu)

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self, X: np.ndarray, p: StudentTHMMParams):
        """
        Run forward-backward and compute:
          gamma  : (T, K)  posterior state probabilities
          xi     : (T-1, K, K)  posterior transition probabilities
          log_lik: float   total log-likelihood of sequence
        """
        log_pi = np.log(np.maximum(p.pi, 1e-300))
        log_A  = np.log(np.maximum(p.A,  1e-300))
        log_obs = _compute_log_obs(X, p.mu, p.sigma, p.nu)

        alpha, log_c = _forward(log_pi, log_A, log_obs)
        beta         = _backward(log_A, log_obs, log_c)

        # Posterior state marginals γ_tk  (log-space then exp)
        log_gamma = alpha + beta
        log_gamma -= np.array([_logsumexp(log_gamma[t]) for t in range(X.shape[0])]).reshape(-1, 1)
        gamma = np.exp(log_gamma)
        gamma = np.maximum(gamma, 0.0)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # Posterior transition counts ξ_{t,ij}
        T, K = X.shape[0], self.n_states
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (
                        alpha[t, i]
                        + log_A[i, j]
                        + log_obs[t + 1, j]
                        + beta[t + 1, j]
                    )
            # Normalise row
            xi_t_flat = xi[t].flatten()
            lse = _logsumexp(xi_t_flat)
            xi[t] = np.exp(xi_t_flat.reshape(K, K) - lse)

        log_lik = float(np.sum(log_c))
        return gamma, xi, log_lik

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray,
                p: StudentTHMMParams) -> StudentTHMMParams:
        """Update all parameters given posteriors."""
        T, d = X.shape
        K = self.n_states

        # Initial state
        pi_new = gamma[0].copy()
        pi_new /= pi_new.sum()

        # Transition matrix
        A_new = xi.sum(axis=0)  # (K, K)
        A_new = np.maximum(A_new, 1e-300)
        A_new /= A_new.sum(axis=1, keepdims=True)

        mu_new    = np.zeros((K, d))
        sigma_new = np.zeros((K, d))
        nu_new    = p.nu.copy()

        for k in range(K):
            gk  = gamma[:, k]          # (T,)
            Nk  = gk.sum()
            if Nk < 1e-10:
                mu_new[k]    = p.mu[k]
                sigma_new[k] = p.sigma[k]
                continue

            # Auxiliary weights w_tk for Student-t ECME
            wk = _compute_aux_weights(X, p.mu[k], p.sigma[k], p.nu[k])  # (T,)
            gwk = gk * wk  # weighted responsibilities

            # Weighted mean
            mu_new[k] = np.sum(gwk[:, None] * X, axis=0) / gwk.sum()

            # Weighted variance (diagonal)
            diff = X - mu_new[k]
            sigma_new[k] = np.sqrt(
                np.sum(gwk[:, None] * diff ** 2, axis=0) / gwk.sum()
            )
            sigma_new[k] = np.maximum(sigma_new[k], 1e-6)

            # Degrees of freedom via Newton-Raphson (unless fixed)
            if not self.fix_nu:
                nu_new[k] = _update_nu(p.nu[k], gk, wk, d)

        return StudentTHMMParams(
            pi=pi_new, A=A_new, mu=mu_new, sigma=sigma_new, nu=nu_new
        )

    # ------------------------------------------------------------------
    # Canonical label ordering  (FR-02)
    # ------------------------------------------------------------------

    def _sort_states(self, p: StudentTHMMParams) -> StudentTHMMParams:
        """
        Reorder states by ascending L2-norm of σ_k.
        State 0 → lowest-volatility regime.
        """
        order = np.argsort(np.linalg.norm(p.sigma, axis=1))
        return StudentTHMMParams(
            pi    = p.pi[order],
            A     = p.A[np.ix_(order, order)],
            mu    = p.mu[order],
            sigma = p.sigma[order],
            nu    = p.nu[order],
            log_likelihood = p.log_likelihood,
            n_iter         = p.n_iter,
            converged      = p.converged,
        )

    # ------------------------------------------------------------------
    # AIC / BIC  (FR-03)
    # ------------------------------------------------------------------

    def _n_params(self, d: int) -> int:
        """
        Free parameter count:
          (K-1)       initial distribution
          K(K-1)      transition matrix (each row sums to 1)
          K*d         means
          K*d         scales (sigmas)
          K           degrees of freedom  (dropped if fix_nu=True)
        """
        K = self.n_states
        return (K - 1) + K * (K - 1) + 2 * K * d + (0 if self.fix_nu else K)

    def aic(self, log_lik: float, d: int) -> float:
        return 2.0 * self._n_params(d) - 2.0 * log_lik

    def bic(self, log_lik: float, T: int, d: int) -> float:
        return self._n_params(d) * np.log(T) - 2.0 * log_lik

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "StudentTHMM":
        """
        Fit the Student-t HMM via Baum-Welch EM.

        Parameters
        ----------
        X : (T, d) array of observations.

        Returns
        -------
        self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.asarray(X, dtype=float)
        T, d = X.shape

        p = self._init_params(X)
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            gamma, xi, log_lik = self._e_step(X, p)
            p = self._m_step(X, gamma, xi, p)

            delta = log_lik - prev_ll
            logger.debug(f"iter {iteration:4d}  ll={log_lik:.4f}  Δ={delta:.2e}")

            if iteration > 0 and abs(delta) < self.tol:
                p.converged = True
                p.n_iter = iteration + 1
                break

            prev_ll = log_lik

        else:
            p.n_iter = self.max_iter

        p.log_likelihood = log_lik

        # Canonical state ordering
        p = self._sort_states(p)

        # Set AIC/BIC after sorting (values are order-invariant)
        p.aic = self.aic(log_lik, d)
        p.bic = self.bic(log_lik, T, d)

        self.params_ = p
        self.is_fitted = True

        logger.info(
            f"StudentTHMM fit: K={self.n_states} d={d} T={T} "
            f"ll={log_lik:.4f} iter={p.n_iter} converged={p.converged}"
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior state probabilities via forward-backward.

        Returns
        -------
        gamma : (T, K)
        """
        self._check_fitted()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        gamma, _, _ = self._e_step(X, self.params_)
        return gamma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        MAP state sequence via log-space Viterbi.

        Returns
        -------
        states : (T,) integer array
        """
        self._check_fitted()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._viterbi(X, self.params_)

    def _viterbi(self, X: np.ndarray, p: StudentTHMMParams) -> np.ndarray:
        """Log-space Viterbi trellis with backtracking."""
        T, d = X.shape
        K = self.n_states
        log_A   = np.log(np.maximum(p.A,  1e-300))
        log_obs = _compute_log_obs(X, p.mu, p.sigma, p.nu)

        viterbi  = np.full((T, K), -np.inf)
        backptr  = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(np.maximum(p.pi, 1e-300)) + log_obs[0]

        for t in range(1, T):
            for k in range(K):
                scores = viterbi[t - 1] + log_A[:, k]
                backptr[t, k] = int(np.argmax(scores))
                viterbi[t, k] = scores[backptr[t, k]] + log_obs[t, k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T - 1] = int(np.argmax(viterbi[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]

        return states

    def score(self, X: np.ndarray) -> float:
        """Return total log-likelihood of X."""
        self._check_fitted()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _, _, ll = self._e_step(X, self.params_)
        return ll

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict/score.")

    def get_model_summary(self) -> dict:
        self._check_fitted()
        p = self.params_
        return {
            "n_states": self.n_states,
            "fix_nu": self.fix_nu,
            "log_likelihood": p.log_likelihood,
            "aic": p.aic,
            "bic": p.bic,
            "converged": p.converged,
            "n_iter": p.n_iter,
            "pi": p.pi.tolist(),
            "A": p.A.tolist(),
            "mu": p.mu.tolist(),
            "sigma": p.sigma.tolist(),
            "nu": p.nu.tolist(),
        }


# ---------------------------------------------------------------------------
# Model selection utility  (FR-03)
# ---------------------------------------------------------------------------

def select_n_states(
    X: np.ndarray,
    k_range: List[int] = [2, 3, 4],
    criterion: str = "bic",
    fix_nu: bool = False,
    max_iter: int = 200,
    random_seed: int = 42,
) -> Tuple[int, float]:
    """
    Fit models for K in k_range and return (best_K, criterion_value).

    Parameters
    ----------
    X         : (T, d) observation array
    k_range   : list of K values to try
    criterion : 'bic' or 'aic'
    fix_nu    : passed through to StudentTHMM
    max_iter  : EM iterations per model
    random_seed : reproducibility

    Returns
    -------
    best_k : int
    best_score : float   (lower is better for AIC/BIC)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    T, d = X.shape

    best_k = k_range[0]
    best_score = np.inf

    for k in k_range:
        model = StudentTHMM(
            n_states=k,
            fix_nu=fix_nu,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        model.fit(X)
        score = model.params_.bic if criterion == "bic" else model.params_.aic
        logger.info(f"K={k}  {criterion.upper()}={score:.2f}  ll={model.params_.log_likelihood:.2f}")
        if score < best_score:
            best_score = score
            best_k = k

    logger.info(f"Best K={best_k} by {criterion.upper()}={best_score:.2f}")
    return best_k, best_score
