"""
Reward predictors for the AIPW pseudo-outcomes used by the confidence sequences.

Only the OLS predictor is used to produce the paper figures; the abstract base
class and the sklearn wrapper are kept so that callers can substitute another
sklearn-compatible regressor if they wish.

Usage
-----
>>> from av_policy_selection import OLSRewardPredictor
>>> predictor = OLSRewardPredictor(actions=np.array([0, 1]))
>>> predictor.fit(contexts, actions, rewards)
>>> r_hat = predictor.predict(contexts, actions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression


class RewardPredictor(ABC):
    """Abstract base class. Subclasses implement ``fit`` and ``predict``.

    Instances are callable: ``predictor(contexts, actions)`` ≡ ``.predict(...)``.
    """

    @abstractmethod
    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> "RewardPredictor":
        """Fit the model on observed bandit data."""

    @abstractmethod
    def predict(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict r̂(X_t, A_t) for each (context, action) pair, clipped to [0, 1]."""

    def __call__(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.predict(contexts, actions)


class SklearnRewardPredictor(RewardPredictor):
    """Wraps any sklearn regressor; features are ``[X_t, one_hot(A_t)]``.

    Parameters
    ----------
    estimator : sklearn regressor with ``fit(X, y)`` and ``predict(X)``.
    actions : np.ndarray
        Full action space, used to build the one-hot encoding.
    """

    def __init__(self, estimator, actions: np.ndarray):
        self.estimator = estimator
        self.actions   = np.asarray(actions)
        self._fitted   = False

    def _build_features(
        self, contexts: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        contexts  = np.asarray(contexts)
        actions   = np.asarray(actions)
        X_ctx     = contexts.reshape(len(contexts), -1)
        action_oh = (actions[:, None] == self.actions[None, :]).astype(float)
        return np.hstack([X_ctx, action_oh])

    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> "SklearnRewardPredictor":
        X = self._build_features(contexts, actions)
        self.estimator.fit(X, np.asarray(rewards))
        self._fitted = True
        return self

    def predict(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X = self._build_features(contexts, actions)
        return np.clip(self.estimator.predict(X), 0.0, 1.0)


class OLSRewardPredictor(SklearnRewardPredictor):
    """Ordinary least squares reward predictor — the default used in the paper.

    Wraps ``sklearn.linear_model.LinearRegression`` with ``[X_t, one_hot(A_t)]``
    features; predictions are clipped to [0, 1].
    """

    def __init__(self, actions: np.ndarray):
        super().__init__(LinearRegression(), actions)
