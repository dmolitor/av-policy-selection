"""
Reward predictors for doubly-robust pseudo-outcome estimation.

Any sklearn-compatible regressor can be wrapped via ``SklearnRewardPredictor``.
``OLSRewardPredictor`` provides an out-of-the-box default using ordinary least
squares with one-hot action features.

Usage
-----
>>> from av_policy_selection import OLSRewardPredictor
>>> predictor = OLSRewardPredictor(actions=np.array([0, 1]))
>>> # BanditSimulator will call predictor.fit(contexts, actions, rewards)
>>> # and then predictor.predict(contexts, actions) automatically.

You can substitute any sklearn regressor::

    from sklearn.ensemble import GradientBoostingRegressor
    predictor = SklearnRewardPredictor(
        GradientBoostingRegressor(), actions=np.array([0, 1])
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression


class RewardPredictor(ABC):
    """Abstract base class for reward predictors.

    Subclasses must implement :meth:`fit` and :meth:`predict`.  Instances are
    callable — ``predictor(contexts, actions)`` delegates to :meth:`predict`.
    """

    @abstractmethod
    def fit(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> "RewardPredictor":
        """Fit the reward model on observed bandit data.

        Parameters
        ----------
        contexts : np.ndarray, shape (T,) or (T, d)
        actions : np.ndarray, shape (T,)
        rewards : np.ndarray, shape (T,)
            Observed rewards R_t ∈ [0, 1].

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict r̂(X_t, A_t) for each (context, action) pair.

        Parameters
        ----------
        contexts : np.ndarray, shape (n,) or (n, d)
        actions : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
            Predictions clipped to [0, 1].
        """

    def __call__(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.predict(contexts, actions)


class SklearnRewardPredictor(RewardPredictor):
    """Reward predictor wrapping any sklearn regressor.

    Features passed to the model are the context features concatenated with a
    one-hot encoding of the action:
    ``[X_t, 1(A_t = a_0), 1(A_t = a_1), ..., 1(A_t = a_{K-1})]``

    Parameters
    ----------
    estimator : sklearn regressor
        Any object with ``fit(X, y)`` and ``predict(X)`` methods.
    actions : np.ndarray
        All possible actions — used to build the one-hot encoding.
    """

    def __init__(self, estimator, actions: np.ndarray):
        self.estimator = estimator
        self.actions = np.asarray(actions)
        self._fitted = False

    def _build_features(
        self, contexts: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        contexts = np.asarray(contexts)
        actions = np.asarray(actions)
        X_ctx = contexts.reshape(len(contexts), -1)  # (n, d)
        action_oh = (actions[:, None] == self.actions[None, :]).astype(float)  # (n, K)
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
    """Ordinary least squares reward predictor (default).

    Wraps ``sklearn.linear_model.LinearRegression`` with context + one-hot
    action features.  Predictions are clipped to [0, 1].

    Parameters
    ----------
    actions : np.ndarray
        All possible actions.
    """

    def __init__(self, actions: np.ndarray):
        super().__init__(LinearRegression(), actions)
