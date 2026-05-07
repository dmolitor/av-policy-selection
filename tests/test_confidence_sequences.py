"""Tests for confidence_sequences module."""

import numpy as np
import pytest

from av_policy_selection import (
    BettingConfidenceSequence,
    LILConfidenceSequence,
    PrPLConfidenceInterval,
    PrPLConfidenceSequence,
)
from av_policy_selection.confidence_sequences import _scaled_xi_and_variance


# ---------------------------------------------------------------------------
# Deterministic tests
# ---------------------------------------------------------------------------


def test_lil_ell_formula():
    """Verify ℓ_t formula with known V̄_t."""
    alpha = 0.1
    k = 0.0
    cs = LILConfidenceSequence(alpha=alpha, k=k)

    # Constant phi = 0.5 so xi = 0.5
    phi = np.full(5, 0.5)
    _, _, V_bar = _scaled_xi_and_variance(phi, k, xi_0=0.5)

    ell_expected = 2.0 * np.log(np.log(V_bar) + 1.0) + np.log(
        LILConfidenceSequence._ZETA2_OVER_E / alpha
    )

    # V_t: all xi = 0.5, xi_hat_0 = 0.5
    # At t=1: V_1 = (0.5 - 0.5)^2 = 0; V_bar_1 = 1
    assert V_bar[0] == 1.0
    # ell_1 = 2*log(log(1)+1) + log(1.65/0.1) = 2*0 + log(16.5) ≈ 2.803
    ell_1_expected = np.log(1.65 / 0.1)
    np.testing.assert_allclose(ell_expected[0], ell_1_expected, rtol=1e-10)


def test_lil_lower_clipped():
    """When unclipped bound < 0, L_t = 0."""
    alpha = 0.01  # tight alpha → large ℓ_t → large correction → bound goes negative
    cs = LILConfidenceSequence(alpha=alpha, k=0.0)
    phi = np.full(3, 0.1)  # small phi → small mean, big correction
    L = cs.lower(phi)
    assert np.all(L >= 0.0)
    # With small phi and tight alpha, at early t bounds should be 0
    assert L[0] == 0.0


def test_lil_upper_mirrors_lower():
    """U_t^LIL = 1 - lower(φ_DRU) exactly."""
    rng = np.random.default_rng(42)
    phi_dru = rng.uniform(0, 1, 20)
    cs = LILConfidenceSequence(alpha=0.1, k=0.0)
    U = cs.upper(phi_dru)
    L_from_dru = cs.lower(phi_dru)
    np.testing.assert_array_equal(U, 1.0 - L_from_dru)


def test_betting_lambda_formula():
    """Verify λ against hand computation at T=1."""
    alpha = 0.1
    sigma0_sq = 0.25
    c = 0.5
    k = 0.0
    nu_hat = 0.5
    cs = BettingConfidenceSequence(
        alpha=alpha, k=k, c=c, sigma0_sq=sigma0_sq
    )

    phi = np.array([0.7])
    lam = cs._lambda(nu_hat, phi, t=1)

    # At i=1: xi_1 = 0.7/(1+0)=0.7; xi_bar_1 = min(0.7, 1.0)=0.7
    # sigma_hat_sq_0 = sigma0_sq = 0.25
    # base_lam_1 = sqrt(2*log(1/0.1) / (0.25 * 1 * log(2))) = sqrt(...)
    base = np.sqrt(2.0 * np.log(1.0 / alpha) / (sigma0_sq * 1.0 * np.log(2.0)))
    cap = c / (k + nu_hat)  # 0.5 / 0.5 = 1.0
    expected = min(base, cap)
    np.testing.assert_allclose(lam[0], expected, rtol=1e-10)


def test_betting_product_T1():
    """T=1 product equals 1 + λ_1*(φ_1 - ν̂)."""
    alpha = 0.1
    nu_hat = 0.5
    phi = np.array([0.8])
    cs = BettingConfidenceSequence(alpha=alpha, k=0.0, c=0.5, sigma0_sq=0.25)

    lam = cs._lambda(nu_hat, phi, t=1)
    log_product = cs._log_product_martingale(nu_hat, phi, t=1)
    expected = np.log1p(lam[0] * (phi[0] - nu_hat))
    np.testing.assert_allclose(log_product, expected, rtol=1e-10)


def test_betting_lower_fast_matches_brentq():
    """Batch-bisection lower() agrees with the brentq reference within tol."""
    rng = np.random.default_rng(7)
    phi = rng.uniform(0.2, 0.8, 80)   # avoid values near 0/1 to stay in non-clamped regime
    cs = BettingConfidenceSequence(alpha=0.1, k=0.0, c=0.5, sigma0_sq=0.25, tol=1e-6)
    L_fast = cs.lower(phi)
    L_ref  = cs._lower_brentq(phi)
    np.testing.assert_allclose(L_fast, L_ref, atol=cs.tol * 10,
                               err_msg="Batch bisection diverges from brentq reference")


def test_lil_cs_shrinks():
    """LIL CS width at t≈200 should be smaller than at t≈50 on i.i.d. data."""
    rng = np.random.default_rng(99)
    phi_l = rng.uniform(0.3, 0.7, 200)        # bounded pseudo-outcomes
    phi_u = 1.0 - phi_l                       # mirrored upper-side input

    cs = LILConfidenceSequence(alpha=0.1, k=0.0)
    L, U = cs.bounds(phi_l, phi_u)
    widths = U - L
    assert widths[100:].mean() < widths[:50].mean()



def test_betting_lower_stride_matches_full():
    """lower(phi, stride=s) at stride positions matches full lower(phi) to within tol."""
    rng = np.random.default_rng(17)
    phi = rng.uniform(0.2, 0.8, 40)
    tol = 1e-6
    cs = BettingConfidenceSequence(alpha=0.1, k=0.0, tol=tol)
    L_full = cs.lower(phi)
    for stride in [2, 5, 8]:
        L_stride = cs.lower(phi, stride=stride)
        # Stride positions: indices stride-1, 2*stride-1, ...
        stride_idx = np.arange(stride - 1, len(phi), stride)
        np.testing.assert_allclose(
            L_stride[stride_idx], L_full[stride_idx], atol=tol * 2,
            err_msg=f"stride={stride}: bound at stride positions disagrees with full computation",
        )
        # Non-stride positions must be zero
        mask = np.ones(len(phi), dtype=bool)
        mask[stride_idx] = False
        assert np.all(L_stride[mask] == 0.0), f"stride={stride}: non-stride positions not zero"


def test_prpl_cs_matches_formula():
    """PrPLConfidenceSequence lower bound matches the closed-form proposition:prpl-cs."""
    alpha = 0.10
    k = 0.0
    c = 0.5
    sigma0_sq = 0.25
    xi_0 = 0.5
    rng = np.random.default_rng(0)
    phi = rng.uniform(0.1, 0.9, 20)

    cs = PrPLConfidenceSequence(alpha=alpha, k=k, c=c, sigma0_sq=sigma0_sq, xi_0=xi_0)
    L = cs.lower(phi)

    T = len(phi)
    cap = 1.0 / (1.0 + k)
    xi = phi / (1.0 + k)

    xi_bar = np.minimum(np.cumsum(xi) / np.arange(1, T + 1), cap)
    sq_dev = np.cumsum((xi - xi_bar) ** 2)
    sigma_sq = (sigma0_sq + sq_dev) / (np.arange(1, T + 1) + 1)
    sigma_sq_lag = np.empty(T)
    sigma_sq_lag[0] = sigma0_sq
    sigma_sq_lag[1:] = sigma_sq[:-1]

    t_vals = np.arange(1, T + 1, dtype=float)
    lam = np.minimum(
        np.sqrt(2.0 * np.log(1.0 / alpha) / (sigma_sq_lag * t_vals * np.log(1.0 + t_vals))),
        c,
    )

    xi_hat_lag = np.empty(T)
    xi_hat_lag[0] = xi_0
    xi_hat_lag[1:] = xi_bar[:-1]

    psi_e = -np.log1p(-lam) - lam
    lam_xi = np.cumsum(lam * xi)
    lam_over_k1 = np.cumsum(lam * cap)
    var_penalty = np.cumsum((xi - xi_hat_lag) ** 2 * psi_e)

    L_expected = np.maximum(
        lam_xi / lam_over_k1 - (np.log(1.0 / alpha) + var_penalty) / lam_over_k1,
        0.0,
    )
    np.testing.assert_allclose(L, L_expected, rtol=1e-10)


def test_prpl_cs_upper_mirrors_lower():
    """U_t^PrPl = 1 - lower(phi_dru) exactly."""
    rng = np.random.default_rng(42)
    phi_dru = rng.uniform(0, 1, 30)
    cs = PrPLConfidenceSequence(alpha=0.05, k=0.0)
    U = cs.upper(phi_dru)
    np.testing.assert_array_equal(U, 1.0 - cs.lower(phi_dru))


def test_prpl_cs_nonnegative():
    """L_t^PrPl >= 0 always."""
    rng = np.random.default_rng(5)
    phi = rng.uniform(0, 1, 100)
    cs = PrPLConfidenceSequence(alpha=0.10)
    L = cs.lower(phi)
    assert np.all(L >= 0.0)


def test_prpl_ci_scalar_outputs():
    """PrPLConfidenceInterval.lower() and upper() return scalars in [0,1]."""
    rng = np.random.default_rng(7)
    phi = rng.uniform(0, 1, 50)
    ci = PrPLConfidenceInterval(alpha=0.10)
    lo = ci.lower(phi)
    hi = ci.upper(phi)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_prpl_ci_tighter_than_cs():
    """At t=n, PrPl CI is tighter than PrPl CS (concentrates betting power at n)."""
    rng = np.random.default_rng(99)
    phi = rng.uniform(0.2, 0.8, 100)
    alpha = 0.10
    cs = PrPLConfidenceSequence(alpha=alpha)
    ci = PrPLConfidenceInterval(alpha=alpha)

    L_cs = cs.lower(phi)[-1]
    U_cs = cs.upper(phi)[-1]
    L_ci = ci.lower(phi)
    U_ci = ci.upper(phi)

    # CI at fixed n should be tighter (larger lower, smaller upper) than CS at same n
    assert L_ci >= L_cs, f"PrPl CI lower ({L_ci:.4f}) should be >= CS lower ({L_cs:.4f}) at t=n"
    assert U_ci <= U_cs, f"PrPl CI upper ({U_ci:.4f}) should be <= CS upper ({U_cs:.4f}) at t=n"


def test_prpl_ci_trajectory_shape():
    """lower_trajectory returns shape (T,) and upper_trajectory = 1 - lower_trajectory(phi_dru)."""
    rng = np.random.default_rng(11)
    T = 40
    phi = rng.uniform(0, 1, T)
    ci = PrPLConfidenceInterval(alpha=0.10)
    lt = ci.lower_trajectory(phi)
    ut = ci.upper_trajectory(phi)
    assert lt.shape == (T,)
    assert ut.shape == (T,)
    np.testing.assert_array_equal(ut, 1.0 - ci.lower_trajectory(phi))


def test_prpl_ci_lower_is_max_trajectory():
    """PrPLConfidenceInterval.lower() == max of lower_trajectory()."""
    rng = np.random.default_rng(13)
    phi = rng.uniform(0, 1, 60)
    ci = PrPLConfidenceInterval(alpha=0.10)
    np.testing.assert_allclose(ci.lower(phi), np.max(ci.lower_trajectory(phi)), rtol=1e-12)
