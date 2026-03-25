"""Tests for simulation module."""

import numpy as np
import pytest
from plotnine import ggplot

from av_policy_selection import BanditSimulator, LILConfidenceSequence, iw_pseudo_outcomes
from av_policy_selection.simulation import BanditData


def _make_simple_simulator(rng_seed: int = 42) -> BanditSimulator:
    """Simple binary bandit for testing."""
    rng = np.random.default_rng(rng_seed)

    def context_fn(T):
        return rng.uniform(0, 1, size=T)

    def logging_policy(ctx, act):
        return np.full(len(act), 0.5)

    def target_policy(ctx, act):
        return (act == 1).astype(float)

    def reward_fn(ctx, act):
        return np.clip(ctx.ravel() * act, 0, 1).astype(float)

    return BanditSimulator(
        logging_policy=logging_policy,
        target_policy=target_policy,
        reward_fn=reward_fn,
        context_fn=context_fn,
        actions=np.array([0, 1]),
        true_policy_value=0.5,
        rng=rng,
    )


def test_shapes():
    """All BanditData arrays have correct shapes."""
    T = 50
    sim = _make_simple_simulator()
    data = sim.simulate(T)

    assert data.contexts.shape == (T,)
    assert data.actions.shape == (T,)
    assert data.rewards.shape == (T,)
    assert data.log_probs.shape == (T,)
    assert data.target_probs.shape == (T,)
    assert data.weights.shape == (T,)


def test_weights_ratio():
    """weights == target_probs / log_probs."""
    sim = _make_simple_simulator()
    data = sim.simulate(30)
    np.testing.assert_allclose(data.weights, data.target_probs / data.log_probs)


def test_rewards_bounded():
    """R_t ∈ [0, 1] for all t."""
    sim = _make_simple_simulator()
    data = sim.simulate(100)
    assert np.all(data.rewards >= 0.0)
    assert np.all(data.rewards <= 1.0)


def test_pseudo_outcomes_iw_consistency():
    """simulator.pseudo_outcomes('iw') matches direct iw_pseudo_outcomes call."""
    sim = _make_simple_simulator()
    data = sim.simulate(30)

    phi_l_sim, phi_u_sim = sim.pseudo_outcomes(data, kind="iw")
    phi_l_direct, phi_u_direct = iw_pseudo_outcomes(data.rewards, data.weights)

    np.testing.assert_array_equal(phi_l_sim, phi_l_direct)
    np.testing.assert_array_equal(phi_u_sim, phi_u_direct)


def test_plot_bounds_returns_ggplot():
    """plot_bounds returns a ggplot object with the right structure."""
    sim = _make_simple_simulator()
    data = sim.simulate(50)
    phi_l, phi_u = sim.pseudo_outcomes(data, kind="iw")
    cs = LILConfidenceSequence(alpha=0.1)
    lower, upper = cs.bounds(phi_l, phi_u)

    p = sim.plot_bounds(lower, upper, title="Test bounds")
    assert isinstance(p, ggplot)


def test_plot_coverage_returns_ggplot():
    """plot_coverage returns a ggplot object."""
    sim = _make_simple_simulator()
    result = sim.coverage_experiment(
        T=30,
        n_trials=10,
        cs_factory=lambda a: LILConfidenceSequence(alpha=a),
        alpha=0.1,
        kind="iw",
    )
    p = sim.plot_coverage(result, max_trials=5)
    assert isinstance(p, ggplot)
