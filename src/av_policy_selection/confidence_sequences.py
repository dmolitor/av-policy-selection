"""
Anytime-valid confidence sequences for off-policy inference.

Implements:
  - LILConfidenceSequence: Proposition lil-eb (Proposition 3) — variance-adaptive
    LIL confidence sequence for time-varying policy value ν̄_t.
  - BettingConfidenceSequence: Theorem dr-fixed-policy-value (Theorem 1) —
    doubly-robust betting confidence sequence for fixed policy value ν.

Reference: Luedtke & Soni (2024), "Anytime-Valid Off-Policy Inference for
Contextual Bandits"
"""

import numpy as np
from scipy.optimize import brentq


def _scaled_xi_and_variance(
    phi: np.ndarray, k: float, xi_0: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute scaled outcomes, lagged running mean, and variance process.

    ξ_t = φ_t / (1 + k)
    ξ̂_t = min(mean(ξ_1..t), 1/(1+k))   [non-lagged current mean]
    V_t = sum_{i=1}^t (ξ_i - ξ̂_{i-1})^2  [uses lagged ξ̂, with ξ̂_0 = xi_0]
    V̄_t = max(V_t, 1)

    [eq. variance-process]

    Parameters
    ----------
    phi : np.ndarray, shape (T,)
        Pseudo-outcomes φ_t.
    k : float
        Truncation parameter.
    xi_0 : float
        Initial estimate ξ̂_0 (predictable starting value).

    Returns
    -------
    xi : np.ndarray, shape (T,)
        Scaled outcomes ξ_t = φ_t / (1+k).
    xi_hat : np.ndarray, shape (T,)
        Non-lagged running mean min(mean(ξ_1..t), 1/(1+k)).
    V_bar : np.ndarray, shape (T,)
        Stabilized variance process max(V_t, 1).
    """
    T = len(phi)
    xi = phi / (1.0 + k)
    cap = 1.0 / (1.0 + k)

    # Non-lagged running mean: xi_hat[t] = min(mean(xi[0..t]), cap)
    xi_cumsum = np.cumsum(xi)
    xi_hat = np.minimum(xi_cumsum / np.arange(1, T + 1), cap)

    # Lagged xi_hat: xi_hat_lag[i] = xi_hat[i-1], with xi_hat_lag[0] = xi_0
    xi_hat_lag = np.empty(T)
    xi_hat_lag[0] = xi_0
    xi_hat_lag[1:] = xi_hat[:-1]

    # V_t = cumsum of (xi_i - xi_hat_{i-1})^2
    V = np.cumsum((xi - xi_hat_lag) ** 2)
    V_bar = np.maximum(V, 1.0)

    return xi, xi_hat, V_bar


class LILConfidenceSequence:
    """Variance-adaptive LIL confidence sequence for time-varying policy value.

    Implements Proposition lil-eb (Proposition 3) from Luedtke & Soni (2024).

    The lower bound at time t is:
      L_t^LIL = (k+1) * [mean(ξ_1..t) - sqrt(γ₁²·ℓ_t·V̄_t + γ₂²·ℓ_t²)/t
                          - γ₂·ℓ_t/t] ∨ 0

    where ℓ_t = 2·log(log(V̄_t)+1) + log(ζ(2)/(e·α)) and:
      γ₁² ≈ 2.13, γ₂² ≈ 1.76, γ₂ ≈ 1.33

    The upper bound uses the mirroring trick: U_t^LIL = 1 - lower(φ_DRU).
    [eq. lil-lower, remark mirroring-trick]
    """

    # Constants from proof with η=e, s=2 [proposition lil-eb]
    _GAMMA1_SQ = 2.13   # γ₁² = ((e^{1/4}+e^{-1/4})/√2)²
    _GAMMA2_SQ = 1.76   # γ₂² = ((√e+1)/2)²
    _GAMMA2 = 1.33      # γ₂ = (√e+1)/2
    _ZETA2_OVER_E = 1.65  # ζ(2)/e

    def __init__(self, alpha: float, k: float = 0.0, xi_0: float = 0.5):
        """
        Parameters
        ----------
        alpha : float
            Significance level (miscoverage probability).
        k : float
            Truncation parameter k ≥ 0.
        xi_0 : float
            Initial estimate ξ̂_0 for the variance process.
        """
        self.alpha = alpha
        self.k = k
        self.xi_0 = xi_0

    def lower(self, phi_drl: np.ndarray) -> np.ndarray:
        """Compute lower confidence bound L_t^LIL for all t.

        Parameters
        ----------
        phi_drl : np.ndarray, shape (T,)
            Lower DR pseudo-outcomes φ_DRL.

        Returns
        -------
        np.ndarray, shape (T,)
            Lower bounds L_t^LIL ≥ 0.
        """
        phi_drl = np.asarray(phi_drl, dtype=float)
        T = len(phi_drl)
        xi, _, V_bar = _scaled_xi_and_variance(phi_drl, self.k, self.xi_0)

        t = np.arange(1, T + 1, dtype=float)
        ell = (
            2.0 * np.log(np.log(V_bar) + 1.0)
            + np.log(self._ZETA2_OVER_E / self.alpha)
        )

        xi_mean = np.cumsum(xi) / t

        correction = (
            np.sqrt(self._GAMMA1_SQ * ell * V_bar + self._GAMMA2_SQ * ell**2) / t
            + self._GAMMA2 * ell / t
        )

        L_unclipped = (1.0 + self.k) * (xi_mean - correction)
        return np.maximum(L_unclipped, 0.0)

    def upper(self, phi_dru: np.ndarray) -> np.ndarray:
        """Compute upper confidence bound U_t^LIL for all t.

        Uses the mirroring trick: U_t^LIL = 1 - lower(φ_DRU).
        [remark mirroring-trick]

        Parameters
        ----------
        phi_dru : np.ndarray, shape (T,)
            Upper DR pseudo-outcomes φ_DRU.

        Returns
        -------
        np.ndarray, shape (T,)
            Upper bounds U_t^LIL ≤ 1.
        """
        return 1.0 - self.lower(np.asarray(phi_dru, dtype=float))

    def bounds(
        self, phi_drl: np.ndarray, phi_dru: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute both lower and upper confidence bounds.

        Parameters
        ----------
        phi_drl : np.ndarray, shape (T,)
            Lower DR pseudo-outcomes φ_DRL.
        phi_dru : np.ndarray, shape (T,)
            Upper DR pseudo-outcomes φ_DRU.

        Returns
        -------
        lower : np.ndarray, shape (T,)
        upper : np.ndarray, shape (T,)
        """
        return self.lower(phi_drl), self.upper(phi_dru)


class BettingConfidenceSequence:
    """Doubly-robust betting confidence sequence for fixed policy value.

    Implements Theorem dr-fixed-policy-value (Theorem 1) from Luedtke & Soni
    (2024).

    The lower bound at time t:
      L_t^DR = inf{ν̂ ∈ [0,1] : ∏_{i=1}^t [1 + λ_i^L(ν̂)·(φ_i^DRL - ν̂)] < 1/α}

    The betting strategy λ_i^L is:
      λ_i = min(sqrt(2·log(1/α) / (σ̂²_{i-1}·i·log(1+i))), c/(k+ν̂))

    The upper bound uses mirroring: U_t^DR = 1 - lower_analog(φ_DRU).
    [eq. dr-lower, betting-strategy-prpl, theorem dr-fixed-policy-value]
    """

    def __init__(
        self,
        alpha: float,
        k: float = 0.0,
        c: float = 0.5,
        sigma0_sq: float = 0.25,
        xi_0: float = 0.5,
        tol: float = 1e-6,
    ):
        """
        Parameters
        ----------
        alpha : float
            Significance level (miscoverage probability).
        k : float
            Truncation parameter k ≥ 0.
        c : float
            Truncation scale for betting strategy, c ∈ (0, 1).
        sigma0_sq : float
            Initial variance estimate σ̂²_0.
        xi_0 : float
            Initial estimate ξ̂_0 (used for variance process initialization).
        tol : float
            Tolerance for brentq root-finding.
        """
        self.alpha = alpha
        self.k = k
        self.c = c
        self.sigma0_sq = sigma0_sq
        self.xi_0 = xi_0
        self.tol = tol

    def _lambda(
        self, nu_hat: float, phi: np.ndarray, t: int
    ) -> np.ndarray:
        """Compute betting multipliers λ_i^L(ν̂) for i=1..t.

        ξ_i = φ_i / (1+k)
        ξ̄_i = min(mean(ξ_1..i), 1/(k+1))   [non-lagged]
        σ̂²_i = (σ²_0 + sum_{j=1}^i (ξ_j - ξ̄_j)²) / (i+1)
        base_λ_i = sqrt(2·log(1/α) / (σ̂²_{i-1}·i·log(1+i)))
        λ_i = min(base_λ_i, c/(k+ν̂))

        [eq. betting-strategy-prpl]

        Parameters
        ----------
        nu_hat : float
            Hypothesized policy value ν̂ ∈ [0, 1].
        phi : np.ndarray, shape (T,) or larger
            Pseudo-outcomes (only first t elements used).
        t : int
            Time horizon (1-indexed, inclusive).

        Returns
        -------
        np.ndarray, shape (t,)
            Betting multipliers λ_i for i=1..t.
        """
        phi_t = phi[:t]
        xi = phi_t / (1.0 + self.k)
        cap = 1.0 / (1.0 + self.k)

        # Non-lagged running mean xi_bar_i = min(mean(xi_1..i), cap)
        xi_cumsum = np.cumsum(xi)
        xi_bar = np.minimum(xi_cumsum / np.arange(1, t + 1), cap)

        # sigma_hat_sq_i = (sigma0_sq + sum_{j=1}^i (xi_j - xi_bar_j)^2) / (i+1)
        sq_deviations = np.cumsum((xi - xi_bar) ** 2)
        sigma_hat_sq = (self.sigma0_sq + sq_deviations) / (np.arange(1, t + 1) + 1)

        # Lagged sigma: sigma_hat_sq_{i-1}; at i=1, use sigma0_sq
        sigma_hat_sq_lag = np.empty(t)
        sigma_hat_sq_lag[0] = self.sigma0_sq
        sigma_hat_sq_lag[1:] = sigma_hat_sq[:-1]

        i_vals = np.arange(1, t + 1, dtype=float)
        base_lambda = np.sqrt(
            2.0 * np.log(1.0 / self.alpha)
            / (sigma_hat_sq_lag * i_vals * np.log(1.0 + i_vals))
        )
        max_lambda = self.c / (self.k + nu_hat) if (self.k + nu_hat) > 0 else np.inf
        return np.minimum(base_lambda, max_lambda)

    def _log_product_martingale(
        self, nu_hat: float, phi: np.ndarray, t: int
    ) -> float:
        """Compute log ∏_{i=1}^t [1 + λ_i^L(ν̂)·(φ_i^DRL - ν̂)].

        Uses np.log1p for numerical stability. [eq. dr-lower]

        Parameters
        ----------
        nu_hat : float
            Hypothesized policy value ν̂.
        phi : np.ndarray
            Pseudo-outcomes (first t used).
        t : int
            Time horizon.

        Returns
        -------
        float
            Log of the product martingale.
        """
        lam = self._lambda(nu_hat, phi, t)
        return float(np.sum(np.log1p(lam * (phi[:t] - nu_hat))))

    def _precompute_base_lambda(self, phi: np.ndarray) -> np.ndarray:
        """Precompute the data-dependent (ν̂-independent) base λ_i for i = 0..T-1.

        base_λ_i = sqrt(2·log(1/α) / (σ̂²_{i-1}·(i+1)·log(i+2)))

        This is separated from _lambda so batch bisection can call it once per
        lower() invocation rather than once per brentq evaluation.
        """
        T = len(phi)
        xi = phi / (1.0 + self.k)
        cap = 1.0 / (1.0 + self.k)

        xi_cumsum = np.cumsum(xi)
        xi_bar = np.minimum(xi_cumsum / np.arange(1, T + 1), cap)

        sq_deviations = np.cumsum((xi - xi_bar) ** 2)
        sigma_hat_sq = (self.sigma0_sq + sq_deviations) / (np.arange(1, T + 1) + 1)

        sigma_hat_sq_lag = np.empty(T)
        sigma_hat_sq_lag[0] = self.sigma0_sq
        sigma_hat_sq_lag[1:] = sigma_hat_sq[:-1]

        i_vals = np.arange(1, T + 1, dtype=float)
        return np.sqrt(
            2.0 * np.log(1.0 / self.alpha)
            / (sigma_hat_sq_lag * i_vals * np.log(1.0 + i_vals))
        )

    def lower(self, phi_drl: np.ndarray) -> np.ndarray:
        """Compute lower confidence bound L_t^DR for all t.

        L_t^DR = inf{ν̂ ∈ [0,1] : product_martingale(ν̂, φ[:t]) < 1/α}

        Uses sequential brentq with two key speedups over the naïve implementation:
        (1) base_lambda is precomputed once in O(T) rather than recomputed inside
            every brentq function evaluation (O(T²) saving).
        (2) Warm-start brackets: since L_t is generally non-decreasing, each
            brentq call begins from the previous result rather than 0, reducing
            iterations from ~20 to ~3-5 once the bound has stabilised.
        [eq. dr-lower, theorem dr-fixed-policy-value]

        Parameters
        ----------
        phi_drl : np.ndarray, shape (T,)
            Lower DR pseudo-outcomes φ_DRL.

        Returns
        -------
        np.ndarray, shape (T,)
            Lower bounds L_t^DR.
        """
        phi_drl = np.asarray(phi_drl, dtype=float)
        T = len(phi_drl)
        log_threshold = np.log(1.0 / self.alpha)
        base_lam = self._precompute_base_lambda(phi_drl)
        bounds = np.zeros(T)
        warm_lo = 0.0   # warm-start: L_t >= L_{t-1} in general

        for t in range(1, T + 1):
            def f(nu_hat: float, _t: int = t) -> float:
                ml = self.c / (self.k + nu_hat) if (self.k + nu_hat) > 0 else np.inf
                lm = np.minimum(base_lam[:_t], ml)
                return float(np.sum(np.log1p(lm * (phi_drl[:_t] - nu_hat)))) - log_threshold

            f_warm = f(warm_lo)
            if f_warm <= 0.0:
                # L_t ≤ warm_lo — rare; fall back to searching below warm_lo
                if warm_lo == 0.0 or f(0.0) <= 0.0:
                    bounds[t - 1] = 0.0
                    warm_lo = 0.0
                else:
                    result = brentq(f, 0.0, warm_lo, xtol=self.tol)
                    bounds[t - 1] = result
                    warm_lo = max(0.0, result - self.tol * 2)
                continue

            # f(warm_lo) > 0 → L_t ≥ warm_lo; search [warm_lo, 1]
            if f(1.0) >= 0.0:
                bounds[t - 1] = 1.0
                warm_lo = 1.0
                continue

            result = brentq(f, warm_lo, 1.0, xtol=self.tol)
            bounds[t - 1] = result
            warm_lo = max(0.0, result - self.tol * 2)

        return bounds

    def _lower_brentq(self, phi_drl: np.ndarray) -> np.ndarray:
        """Reference lower() using sequential brentq calls (for testing only)."""
        phi_drl = np.asarray(phi_drl, dtype=float)
        T = len(phi_drl)
        log_threshold = np.log(1.0 / self.alpha)
        bounds = np.zeros(T)

        for t in range(1, T + 1):
            def f(nu_hat: float, _t: int = t) -> float:
                return self._log_product_martingale(nu_hat, phi_drl, _t) - log_threshold

            f0 = f(0.0)
            if f0 <= 0.0:
                bounds[t - 1] = 0.0
                continue
            f1 = f(1.0)
            if f1 >= 0.0:
                bounds[t - 1] = 1.0
                continue
            bounds[t - 1] = brentq(f, 0.0, 1.0, xtol=self.tol)

        return bounds

    def upper(self, phi_dru: np.ndarray) -> np.ndarray:
        """Compute upper confidence bound U_t^DR for all t.

        Uses the mirroring trick: U_t^DR = 1 - lower_analog(φ_DRU).
        [remark mirroring-trick]

        Parameters
        ----------
        phi_dru : np.ndarray, shape (T,)
            Upper DR pseudo-outcomes φ_DRU.

        Returns
        -------
        np.ndarray, shape (T,)
            Upper bounds U_t^DR.
        """
        return 1.0 - self.lower(np.asarray(phi_dru, dtype=float))

    def bounds(
        self, phi_drl: np.ndarray, phi_dru: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute both lower and upper confidence bounds.

        Parameters
        ----------
        phi_drl : np.ndarray, shape (T,)
            Lower DR pseudo-outcomes φ_DRL.
        phi_dru : np.ndarray, shape (T,)
            Upper DR pseudo-outcomes φ_DRU.

        Returns
        -------
        lower : np.ndarray, shape (T,)
        upper : np.ndarray, shape (T,)
        """
        return self.lower(phi_drl), self.upper(phi_dru)




class HoeffdingConfidenceBound:
    """Fixed-sample Hoeffding confidence bound (NOT anytime-valid).

    Implements the width: width_t = sqrt(log(2/α) / (2·t)), applied to
    normalized pseudo-outcomes ξ_t = φ_t / (1+k) ∈ [0, 1].

      L_t = (1+k) · [mean(ξ_1..t) - width_t] ∨ 0
      U_t = 1 − L_t(φ_DRU)   [mirroring trick, same as LIL]

    IMPORTANT: these bounds are valid at any single fixed time t, but NOT
    simultaneously across all t. They are tighter than LIL CSs (no log-log
    factor), yielding faster stopping times at the cost of losing the
    anytime guarantee.

    Parameters
    ----------
    alpha : float
        Significance level.
    k : float
        Truncation parameter k >= 0.
    """

    def __init__(self, alpha: float, k: float = 0.0):
        self.alpha = alpha
        self.k = k

    def lower(self, phi_drl: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi_drl, dtype=float)
        T = len(phi)
        xi = phi / (1.0 + self.k)
        t = np.arange(1, T + 1, dtype=float)
        width = np.sqrt(np.log(2.0 / self.alpha) / (2.0 * t))
        xi_mean = np.cumsum(xi) / t
        return np.maximum((1.0 + self.k) * (xi_mean - width), 0.0)

    def upper(self, phi_dru: np.ndarray) -> np.ndarray:
        return 1.0 - self.lower(np.asarray(phi_dru, dtype=float))

    def bounds(self, phi_drl, phi_dru):
        return self.lower(phi_drl), self.upper(phi_dru)
