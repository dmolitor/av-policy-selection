"""Tests for pseudo_outcomes module."""

import numpy as np
import pytest

from av_policy_selection import (
    dr_pseudo_outcomes,
    importance_weights,
    iw_pseudo_outcomes,
)


def test_importance_weights():
    pi = np.array([0.3, 0.6, 0.9])
    h = np.array([0.5, 0.5, 0.5])
    w = importance_weights(pi, h)
    expected = np.array([0.6, 1.2, 1.8])
    np.testing.assert_allclose(w, expected)


def test_iw_outcomes():
    rewards = np.array([0.0, 0.5, 1.0])
    weights = np.array([2.0, 1.0, 0.5])
    phi_l, phi_u = iw_pseudo_outcomes(rewards, weights)
    np.testing.assert_allclose(phi_l, weights * rewards)
    np.testing.assert_allclose(phi_u, weights * (1.0 - rewards))


def test_dr_k0_equals_iw():
    """DR with k=0, r̂=0 should equal IW pseudo-outcomes."""
    rng = np.random.default_rng(42)
    T = 10
    rewards = rng.uniform(0, 1, T)
    weights = rng.uniform(0.5, 2.0, T)
    r_hat = np.zeros(T)
    r_hat_pi_mean_lower = np.zeros(T)
    r_hat_pi_mean_upper = np.zeros(T)
    k = 0.0

    phi_dr_l, phi_dr_u = dr_pseudo_outcomes(
        rewards, weights, r_hat, r_hat_pi_mean_lower, r_hat_pi_mean_upper, k
    )
    phi_iw_l, phi_iw_u = iw_pseudo_outcomes(rewards, weights)

    np.testing.assert_allclose(phi_dr_l, phi_iw_l)
    np.testing.assert_allclose(phi_dr_u, phi_iw_u)


def test_dr_formula_single_step():
    """Verify DR pseudo-outcomes with hand-computed single-step values."""
    R = np.array([0.8])
    w = np.array([1.5])
    r_hat = np.array([0.6])
    k = 0.4

    # threshold = k/w = 0.4/1.5 ≈ 0.2667
    # min(r_hat, threshold) = min(0.6, 0.2667) = 0.2667
    # phi_DRL = 1.5*(0.8 - 0.2667) + r_hat_pi_mean_lower
    threshold = k / w[0]
    r_hat_pi_mean_lower = np.array([0.2])  # caller-provided
    r_hat_pi_mean_upper = np.array([0.15])

    phi_l_expected = w[0] * (R[0] - min(r_hat[0], threshold)) + r_hat_pi_mean_lower[0]
    phi_u_expected = (
        w[0] * ((1 - R[0]) - min(1 - r_hat[0], threshold)) + r_hat_pi_mean_upper[0]
    )

    phi_l, phi_u = dr_pseudo_outcomes(
        R, w, r_hat, r_hat_pi_mean_lower, r_hat_pi_mean_upper, k
    )
    np.testing.assert_allclose(phi_l[0], phi_l_expected, rtol=1e-10)
    np.testing.assert_allclose(phi_u[0], phi_u_expected, rtol=1e-10)


def test_dr_lower_bounded():
    """φ_DRL >= -k always (mathematical invariant)."""
    rng = np.random.default_rng(7)
    T = 50
    rewards = rng.uniform(0, 1, T)
    weights = rng.uniform(0.5, 3.0, T)
    r_hat = rng.uniform(0, 1, T)
    k = 0.5
    threshold = k / weights

    # Compute r_hat_pi_mean_lower correctly: E_{a~π}[min(r̂_a, k/w_t)]
    # Use binary actions with pi(1|x)=0.4
    pi = 0.4
    r_hat_0 = rng.uniform(0, 1, T)
    r_hat_1 = rng.uniform(0, 1, T)
    r_hat_pi_mean_lower = pi * np.minimum(r_hat_1, threshold) + (1 - pi) * np.minimum(
        r_hat_0, threshold
    )
    r_hat_pi_mean_upper = pi * np.minimum(1 - r_hat_1, threshold) + (
        1 - pi
    ) * np.minimum(1 - r_hat_0, threshold)

    phi_l, _ = dr_pseudo_outcomes(
        rewards, weights, r_hat, r_hat_pi_mean_lower, r_hat_pi_mean_upper, k
    )
    assert np.all(phi_l >= -k - 1e-10), f"min(phi_l) = {phi_l.min()}, expected >= {-k}"
