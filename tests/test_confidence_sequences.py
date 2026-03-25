"""Tests for confidence_sequences module."""

import numpy as np
import pytest

from av_policy_selection import (
    BanditSimulator,
    BettingConfidenceSequence,
    HoeffdingConfidenceBound,
    LILConfidenceSequence,
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


# ---------------------------------------------------------------------------
# Stochastic (coverage) tests
# ---------------------------------------------------------------------------


def _make_bernoulli_simulator(rng_seed: int = 0) -> BanditSimulator:
    """Binary bandit: X~Bernoulli(0.5), two actions {0,1}.

    Logging: uniform (p=0.5 each action).
    Target: always take action 1 (π(1|x)=1).
    Reward: R = X·A (only action 1 in context X=1 pays off).
    True policy value: ν = E_π[R] = E[X·1] = 0.5.
    """
    rng = np.random.default_rng(rng_seed)

    def context_fn(T):
        return rng.integers(0, 2, size=T).astype(float)

    def logging_policy(ctx, act):
        return np.full(len(act), 0.5)

    def target_policy(ctx, act):
        # π(1|x) = 1, π(0|x) = 0
        return (act == 1).astype(float)

    def reward_fn(ctx, act):
        return (ctx * act).astype(float)

    def reward_predictor(ctx, act):
        # r̂(x, a) = x * a  (perfect model)
        return (ctx.ravel() * act).astype(float)

    return BanditSimulator(
        logging_policy=logging_policy,
        target_policy=target_policy,
        reward_fn=reward_fn,
        context_fn=context_fn,
        actions=np.array([0, 1]),
        true_policy_value=0.5,
        reward_predictor=reward_predictor,
        k=0.0,
        rng=rng,
    )


@pytest.mark.slow
def test_lil_coverage_bernoulli():
    """LIL CS should cover ν for ≥ 1-α-0.03 of trials."""
    alpha = 0.1
    sim = _make_bernoulli_simulator(seed := 123)
    result = sim.coverage_experiment(
        T=200,
        n_trials=500,
        cs_factory=lambda a: LILConfidenceSequence(alpha=a, k=0.0),
        alpha=alpha,
        kind="iw",
    )
    assert result["coverage_rate"] >= (1 - alpha) - 0.03, (
        f"LIL coverage {result['coverage_rate']:.3f} < {1-alpha-0.03:.3f}"
    )


@pytest.mark.slow
def test_betting_coverage_bernoulli():
    """Betting CS should cover ν for ≥ 1-α-0.03 of trials."""
    alpha = 0.1
    sim = _make_bernoulli_simulator(seed := 456)
    result = sim.coverage_experiment(
        T=200,
        n_trials=500,
        cs_factory=lambda a: BettingConfidenceSequence(alpha=a, k=0.0),
        alpha=alpha,
        kind="iw",
    )
    assert result["coverage_rate"] >= (1 - alpha) - 0.03, (
        f"Betting coverage {result['coverage_rate']:.3f} < {1-alpha-0.03:.3f}"
    )


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
    """Average CI width at T=200 should be less than at T=50."""
    rng = np.random.default_rng(99)

    def context_fn(T):
        return rng.integers(0, 2, size=T).astype(float)

    def logging_policy(ctx, act):
        return np.full(len(act), 0.5)

    def target_policy(ctx, act):
        return (act == 1).astype(float)

    def reward_fn(ctx, act):
        return (ctx * act).astype(float)

    sim = BanditSimulator(
        logging_policy=logging_policy,
        target_policy=target_policy,
        reward_fn=reward_fn,
        context_fn=context_fn,
        actions=np.array([0, 1]),
        true_policy_value=0.5,
        rng=rng,
    )

    data = sim.simulate(200)
    phi_l, phi_u = sim.pseudo_outcomes(data, kind="iw")

    cs = LILConfidenceSequence(alpha=0.1, k=0.0)
    L, U = cs.bounds(phi_l, phi_u)
    widths = U - L

    avg_width_50 = widths[:50].mean()
    avg_width_200 = widths[100:].mean()
    assert avg_width_200 < avg_width_50, (
        f"Expected shrinkage: width at t~200 ({avg_width_200:.4f}) "
        f">= width at t~50 ({avg_width_50:.4f})"
    )



def test_hoeffding_lower_matches_formula():
    alpha = 0.10
    k = 0.0
    cb = HoeffdingConfidenceBound(alpha=alpha, k=k)
    phi = np.full(10, 0.5)
    L = cb.lower(phi)
    for t_idx, t in [(0, 1), (4, 5), (9, 10)]:
        xi_mean = 0.5
        width = np.sqrt(np.log(2.0 / alpha) / (2.0 * t))
        expected = max((xi_mean - width), 0.0)
        np.testing.assert_allclose(L[t_idx], expected, rtol=1e-10)


def test_hoeffding_lower_clipped():
    alpha = 0.001
    cb = HoeffdingConfidenceBound(alpha=alpha, k=0.0)
    phi = np.full(5, 0.05)
    L = cb.lower(phi)
    assert L[0] == 0.0
    assert np.all(L >= 0.0)


def test_hoeffding_upper_mirrors_lower():
    rng = np.random.default_rng(42)
    phi_dru = rng.uniform(0, 2, 20)
    cb = HoeffdingConfidenceBound(alpha=0.05, k=1.0)
    U = cb.upper(phi_dru)
    L_from_lower = cb.lower(phi_dru)
    np.testing.assert_array_equal(U, 1.0 - L_from_lower)
