"""Tests for the OLS / Sklearn reward predictors used by the AIPW pseudo-outcomes."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from av_policy_selection import OLSRewardPredictor, SklearnRewardPredictor
from av_policy_selection.reward_predictors import RewardPredictor


def _make_data(T: int = 50, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    contexts = rng.uniform(0, 1, T)
    actions  = rng.integers(0, 2, T)
    rewards  = np.clip(contexts * actions + 0.1 * rng.standard_normal(T), 0, 1)
    return contexts, actions, rewards


def test_ols_is_reward_predictor():
    assert isinstance(OLSRewardPredictor(actions=np.array([0, 1])), RewardPredictor)


def test_ols_fit_predict_shapes():
    contexts, actions, rewards = _make_data()
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds = predictor.predict(contexts, actions)
    assert preds.shape == (len(contexts),)


def test_ols_predictions_clipped_to_unit_interval():
    contexts, actions, rewards = _make_data()
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds = predictor.predict(contexts, actions)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_ols_callable_interface_matches_predict():
    contexts, actions, rewards = _make_data(T=20)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    np.testing.assert_array_equal(
        predictor(contexts, actions), predictor.predict(contexts, actions)
    )


def test_ols_predict_before_fit_raises():
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    with pytest.raises(RuntimeError, match="fit"):
        predictor.predict(np.array([0.5]), np.array([1]))


def test_sklearn_wrapper_with_ridge_estimator():
    contexts, actions, rewards = _make_data()
    predictor = SklearnRewardPredictor(Ridge(alpha=1.0), actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    preds = predictor.predict(contexts, actions)
    assert preds.shape == (len(contexts),)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_ols_handles_2d_contexts():
    rng = np.random.default_rng(7)
    T = 30
    contexts = rng.uniform(0, 1, (T, 3))
    actions  = rng.integers(0, 2, T)
    rewards  = np.clip(contexts[:, 0] * actions, 0, 1)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(contexts, actions, rewards)
    assert predictor.predict(contexts, actions).shape == (T,)
