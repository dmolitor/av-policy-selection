"""Tests for reward_predictors module."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from av_policy_selection import OLSRewardPredictor, SklearnRewardPredictor
from av_policy_selection.reward_predictors import RewardPredictor


def _make_data(T: int = 50, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    contexts = rng.uniform(0, 1, T)
    actions = rng.integers(0, 2, T)
    rewards = np.clip(contexts * actions + 0.1 * rng.standard_normal(T), 0, 1)
    return contexts, actions, rewards


def test_ols_is_reward_predictor():
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    assert isinstance(predictor, RewardPredictor)


def test_ols_fit_predict_shapes():
    contexts, actions, rewards = _make_data()
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds = predictor.predict(contexts, actions)
    assert preds.shape == (len(contexts),)


def test_ols_predictions_clipped():
    """Predictions must be in [0, 1]."""
    contexts, actions, rewards = _make_data()
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds = predictor.predict(contexts, actions)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_ols_callable_interface():
    """Predictor is callable via __call__."""
    contexts, actions, rewards = _make_data(T=20)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds_call = predictor(contexts, actions)
    preds_predict = predictor.predict(contexts, actions)
    np.testing.assert_array_equal(preds_call, preds_predict)


def test_ols_predict_before_fit_raises():
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    with pytest.raises(RuntimeError, match="fit"):
        predictor.predict(np.array([0.5]), np.array([1]))


def test_sklearn_custom_estimator():
    """SklearnRewardPredictor works with an arbitrary sklearn estimator."""
    contexts, actions, rewards = _make_data()
    predictor = SklearnRewardPredictor(Ridge(alpha=1.0), actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds = predictor.predict(contexts, actions)
    assert preds.shape == (len(contexts),)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_ols_2d_contexts():
    """Predictor handles 2D context arrays."""
    rng = np.random.default_rng(7)
    T = 30
    contexts = rng.uniform(0, 1, (T, 3))
    actions = rng.integers(0, 2, T)
    rewards = np.clip(contexts[:, 0] * actions, 0, 1)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds = predictor.predict(contexts, actions)
    assert preds.shape == (T,)


def test_default_ols_in_simulator():
    """BanditSimulator with reward_predictor=None uses OLS for DR outcomes."""
    from av_policy_selection import BanditSimulator

    rng = np.random.default_rng(42)

    def context_fn(T):
        return rng.uniform(0, 1, T)

    def logging_policy(ctx, act):
        return np.full(len(act), 0.5)

    def target_policy(ctx, act):
        return (act == 1).astype(float)

    def reward_fn(ctx, act):
        return np.clip(ctx.ravel() * act, 0, 1).astype(float)

    sim = BanditSimulator(
        logging_policy=logging_policy,
        target_policy=target_policy,
        reward_fn=reward_fn,
        context_fn=context_fn,
        actions=np.array([0, 1]),
        true_policy_value=0.5,
        reward_predictor=None,  # default OLS
        rng=rng,
    )
    data = sim.simulate(40)
    phi_l, phi_u = sim.pseudo_outcomes(data, kind="dr")
    assert phi_l.shape == (40,)
    assert phi_u.shape == (40,)


def test_sklearn_predictor_in_simulator():
    """BanditSimulator accepts a SklearnRewardPredictor."""
    from sklearn.ensemble import GradientBoostingRegressor

    from av_policy_selection import BanditSimulator

    rng = np.random.default_rng(99)

    def context_fn(T):
        return rng.uniform(0, 1, T)

    def logging_policy(ctx, act):
        return np.full(len(act), 0.5)

    def target_policy(ctx, act):
        return (act == 1).astype(float)

    def reward_fn(ctx, act):
        return np.clip(ctx.ravel() * act, 0, 1).astype(float)

    predictor = SklearnRewardPredictor(
        GradientBoostingRegressor(n_estimators=10, random_state=0),
        actions=np.array([0, 1]),
    )
    sim = BanditSimulator(
        logging_policy=logging_policy,
        target_policy=target_policy,
        reward_fn=reward_fn,
        context_fn=context_fn,
        actions=np.array([0, 1]),
        true_policy_value=0.5,
        reward_predictor=predictor,
        rng=rng,
    )
    data = sim.simulate(40)
    phi_l, phi_u = sim.pseudo_outcomes(data, kind="dr")
    assert phi_l.shape == (40,)
    assert phi_u.shape == (40,)
