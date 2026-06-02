"""
Anytime-valid confidence sequences for off-policy inference.

Implements from Waudby-Smith et al.:
  - LILConfidenceSequence: Proposition lil-eb (Proposition 3) — variance-adaptive
    LIL confidence sequence for time-varying policy value ν̄_t.
  - BettingConfidenceSequence: Theorem dr-fixed-policy-value (Theorem 1) —
    doubly-robust betting confidence sequence for fixed policy value ν.
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

    Implements Proposition lil-eb (Proposition 3) of Waudby-Smith et al.

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


class PrPLConfidenceSequence:
    """Closed-form predictable plug-in confidence sequence for fixed ν.

    Implements Proposition prpl-cs. The lower bound is:

      L_t^PrPl = (Σλ_i·ξ_i / Σλ_i/(k+1)
                  - (log(1/α) + Σ(ξ_i−ξ̂_{i-1})²·ψ_E(λ_i)) / Σλ_i/(k+1)) ∨ 0

    where ψ_E(λ) = −log(1−λ) − λ, and the n-independent (CS) tuning is:

      λ_t = sqrt(2·log(1/α) / (σ̂²_{t-1}·t·log(1+t))) ∧ c

    Two means are used:
      ξ̄_t (non-lagged) = min(mean(ξ_1..t), cap)  — for σ̂² only
      ξ̂_{t-1} (predictable) = min(mean(ξ_1..t-1), cap)  — in the bound formula

    The upper bound uses the mirroring trick: U_t^PrPl = 1 − lower(φ_DRU).
    """

    def __init__(
        self,
        alpha: float,
        k: float = 0.0,
        c: float = 0.5,
        sigma0_sq: float = 0.25,
        xi_0: float = 0.5,
    ):
        """
        Parameters
        ----------
        alpha : float
            Significance level.
        k : float
            Truncation parameter k ≥ 0.
        c : float
            Truncation scale for λ_t, c ∈ (0, 1).
        sigma0_sq : float
            Initial variance estimate σ̂²_0.
        xi_0 : float
            Initial predictable mean ξ̂_0 (prior guess for ξ mean).
        """
        self.alpha = alpha
        self.k = k
        self.c = c
        self.sigma0_sq = sigma0_sq
        self.xi_0 = xi_0

    def _compute_bounds(
        self, phi_drl: np.ndarray, n_fixed: int | None = None
    ) -> np.ndarray:
        """Core computation shared by CS and CI.

        Parameters
        ----------
        phi_drl : np.ndarray, shape (T,)
        n_fixed : int or None
            If None: use CS tuning  λ_t ∝ 1/sqrt(t·log(1+t)·σ̂²_{t-1}).
            If int:  use CI tuning  λ_t ∝ 1/sqrt(n·σ̂²_{t-1})  (n = n_fixed).

        Returns
        -------
        np.ndarray, shape (T,)
        """
        phi = np.asarray(phi_drl, dtype=float)
        T = len(phi)
        cap = 1.0 / (1.0 + self.k)
        xi = phi / (1.0 + self.k)

        # ξ̄_t (non-lagged): for σ̂² computation only [eq:prplcs]
        xi_bar = np.minimum(np.cumsum(xi) / np.arange(1, T + 1), cap)

        # σ̂²_t = (σ₀² + Σ(ξ_i − ξ̄_i)²) / (t+1)  [eq:betting-strategy-prpl-sigmahat2]
        sq_dev = np.cumsum((xi - xi_bar) ** 2)
        sigma_sq = (self.sigma0_sq + sq_dev) / (np.arange(1, T + 1) + 1)

        sigma_sq_lag = np.empty(T)
        sigma_sq_lag[0] = self.sigma0_sq
        sigma_sq_lag[1:] = sigma_sq[:-1]

        # λ_t: CS uses t·log(1+t); CI uses fixed n [eq:prplcs, corollary:prpl-ci]
        t_vals = np.arange(1, T + 1, dtype=float)
        if n_fixed is None:
            denom = sigma_sq_lag * t_vals * np.log(1.0 + t_vals)
        else:
            denom = sigma_sq_lag * float(n_fixed)
        lam = np.minimum(np.sqrt(2.0 * np.log(1.0 / self.alpha) / denom), self.c)

        # ξ̂_{t-1} (predictable lagged mean): used directly in the bound [eq:prplcs]
        xi_hat_lag = np.empty(T)
        xi_hat_lag[0] = self.xi_0
        xi_hat_lag[1:] = xi_bar[:-1]

        # ψ_E(λ) = −log(1−λ) − λ  [line 728, proof:prpl-cs]
        psi_e = -np.log1p(-lam) - lam

        # Cumulative sums for the closed-form bound
        lam_xi = np.cumsum(lam * xi)
        lam_over_k1 = np.cumsum(lam * cap)         # Σλ_i/(k+1) = cap·Σλ_i
        var_penalty = np.cumsum((xi - xi_hat_lag) ** 2 * psi_e)

        log_1_alpha = np.log(1.0 / self.alpha)
        L_unclipped = (
            lam_xi / lam_over_k1
            - (log_1_alpha + var_penalty) / lam_over_k1
        )
        return np.maximum(L_unclipped, 0.0)

    def lower(self, phi_drl: np.ndarray) -> np.ndarray:
        """Lower CS L_t^PrPl for all t=1..T.

        Parameters
        ----------
        phi_drl : np.ndarray, shape (T,)

        Returns
        -------
        np.ndarray, shape (T,)
        """
        return self._compute_bounds(np.asarray(phi_drl, dtype=float), n_fixed=None)

    def upper(self, phi_dru: np.ndarray) -> np.ndarray:
        """Upper CS U_t^PrPl = 1 − lower(φ_DRU) for all t=1..T.

        [remark:mirroring-trick]
        """
        return 1.0 - self.lower(np.asarray(phi_dru, dtype=float))

    def bounds(
        self, phi_drl: np.ndarray, phi_dru: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.lower(phi_drl), self.upper(phi_dru)


class PrPLConfidenceInterval:
    """Fixed-n PrPl confidence interval for fixed ν.

    Implements Corollary prpl-ci. Identical to
    PrPLConfidenceSequence but uses n-tuned betting parameters:

      dot_λ_{t,n} = sqrt(2·log(1/α) / (n·σ̂²_{t-1})) ∧ c

    concentrating betting power at the planned horizon n. The valid (1−α) CI is:

      dot_L_n^PrPl = max_{1≤t≤n} L_t^PrPl,   dot_U_n^PrPl = min_{1≤t≤n} U_t^PrPl

    lower() / upper() return these scalars. lower_trajectory() / upper_trajectory()
    return the full (n,) path, which is NOT anytime-valid and fails under peeking.
    """

    def __init__(
        self,
        alpha: float,
        k: float = 0.0,
        c: float = 0.5,
        sigma0_sq: float = 0.25,
        xi_0: float = 0.5,
    ):
        self.alpha = alpha
        self.k = k
        self.c = c
        self.sigma0_sq = sigma0_sq
        self.xi_0 = xi_0
        self._cs = PrPLConfidenceSequence(alpha, k, c, sigma0_sq, xi_0)

    def lower_trajectory(self, phi_drl: np.ndarray) -> np.ndarray:
        """L_t^PrPl trajectory t=1..n with n-tuned λ (NOT anytime-valid).

        Returns
        -------
        np.ndarray, shape (n,)
        """
        phi = np.asarray(phi_drl, dtype=float)
        return self._cs._compute_bounds(phi, n_fixed=len(phi))

    def upper_trajectory(self, phi_dru: np.ndarray) -> np.ndarray:
        """U_t^PrPl = 1 − lower_trajectory(φ_DRU) (NOT anytime-valid)."""
        return 1.0 - self.lower_trajectory(np.asarray(phi_dru, dtype=float))

    def lower(self, phi_drl: np.ndarray) -> float:
        """Valid (1−α) lower CI: dot_L_n = max_t L_t^PrPl.

        [corollary:prpl-ci, eq:prpl-ci]
        """
        return float(np.max(self.lower_trajectory(phi_drl)))

    def upper(self, phi_dru: np.ndarray) -> float:
        """Valid (1−α) upper CI: dot_U_n = min_t U_t^PrPl."""
        return float(np.min(self.upper_trajectory(phi_dru)))

    def bounds(self, phi_drl: np.ndarray, phi_dru: np.ndarray) -> tuple[float, float]:
        return self.lower(phi_drl), self.upper(phi_dru)


class BettingConfidenceSequence:
    """Doubly-robust betting confidence sequence for fixed policy value.

    Implements Theorem dr-fixed-policy-value (Theorem 1)

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
        tol: float = 1e-3,
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
            Tolerance for brentq root-finding.  1e-3 is sufficient for
            stopping-time detection (policy gaps >> 1e-3); use 1e-6 only
            when high-precision bound values are needed (e.g. unit tests).
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

    def lower(self, phi_drl: np.ndarray, stride: int = 1) -> np.ndarray:
        """Compute lower confidence bound L_t^DR for all t (or every stride-th t).

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
        stride : int
            Compute bounds only at t = stride, 2·stride, …, T.  All other
            positions are left at 0.  stride=1 (default) computes every t.
            Warm-start carries over correctly between stride points since
            L_t is non-decreasing; the bound at each computed t is identical
            to the full-stride computation.  Useful for visualisation when
            only a coarse time grid is needed, giving a stride× speedup.

        Returns
        -------
        np.ndarray, shape (T,)
            Lower bounds L_t^DR (0 at non-stride positions when stride > 1).
        """
        phi_drl = np.asarray(phi_drl, dtype=float)
        T = len(phi_drl)
        log_threshold = np.log(1.0 / self.alpha)
        base_lam = self._precompute_base_lambda(phi_drl)

        bounds = np.zeros(T)
        warm_lo = 0.0

        for t in range(stride, T + 1, stride):
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

    def upper(self, phi_dru: np.ndarray, stride: int = 1) -> np.ndarray:
        """Compute upper confidence bound U_t^DR for all t (or every stride-th t).

        Uses the mirroring trick: U_t^DR = 1 - lower_analog(φ_DRU).
        [remark mirroring-trick]

        Parameters
        ----------
        phi_dru : np.ndarray, shape (T,)
            Upper DR pseudo-outcomes φ_DRU.
        stride : int
            Passed through to lower(); see lower() for details.

        Returns
        -------
        np.ndarray, shape (T,)
            Upper bounds U_t^DR (1 at non-stride positions when stride > 1).
        """
        return 1.0 - self.lower(np.asarray(phi_dru, dtype=float), stride=stride)

    def bounds(
        self, phi_drl: np.ndarray, phi_dru: np.ndarray, stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute both lower and upper confidence bounds.

        Parameters
        ----------
        phi_drl : np.ndarray, shape (T,)
            Lower DR pseudo-outcomes φ_DRL.
        phi_dru : np.ndarray, shape (T,)
            Upper DR pseudo-outcomes φ_DRU.
        stride : int
            Passed through to lower()/upper(); see lower() for details.

        Returns
        -------
        lower : np.ndarray, shape (T,)
        upper : np.ndarray, shape (T,)
        """
        return self.lower(phi_drl, stride=stride), self.upper(phi_dru, stride=stride)


