"""
Anytime-valid optimal policy identification.

Implements the optimal policy set S_t and stopping time τ from:
    resources/anytime-valid-optimal-policy-identification.tex
"""

from __future__ import annotations

import numpy as np


class PolicySelector:
    """Anytime-valid optimal policy identification.

    Given pre-computed (m, T) lower and upper confidence-sequence bound arrays
    (with the α/m union-bound correction already applied during CS construction),
    computes the optimal policy set S_t and the stopping time τ.
    """

    @staticmethod
    def optimal_set(
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        """Compute the optimal policy set S_t for each time step.

        A policy π_i belongs to S_t when its upper confidence bound U_t(π_i)
        is at least as large as the maximum lower confidence bound across all
        policies::

            S_t = {π ∈ Π : U_t(π; α/m) ≥ max_{π'∈Π} L_t(π'; α/m)}

        Parameters
        ----------
        lower_bounds : np.ndarray, shape (m, T)
            L_t(π_i; α/m) for each policy i and time t.
        upper_bounds : np.ndarray, shape (m, T)
            U_t(π_i; α/m) for each policy i and time t.

        Returns
        -------
        np.ndarray, shape (m, T), dtype bool
            ``True`` at (i, t) iff π_i ∈ S_t.
        """
        if lower_bounds.ndim != 2 or upper_bounds.ndim != 2:
            raise ValueError("lower_bounds and upper_bounds must be 2D arrays.")
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError(
                f"Shape mismatch: lower_bounds {lower_bounds.shape} vs "
                f"upper_bounds {upper_bounds.shape}."
            )

        # max_{π'∈Π} L_t(π') — shape (T,)
        max_lower = lower_bounds.max(axis=0)
        # π_i ∈ S_t iff U_t(π_i) ≥ max_lower_t
        return upper_bounds >= max_lower[np.newaxis, :]

    @staticmethod
    def stopping_time(
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> int:
        """Compute the stopping time τ.

        τ is the first time at which some policy π' has a lower bound strictly
        greater than the maximum upper bound of every other policy::

            τ = inf{t ≥ 1 : ∃π' s.t. L_t(π'; α/m) > max_{π≠π'} U_t(π; α/m)}

        When m=1 there is no competitor, so τ is defined as T+1 (never stops).
        If τ is never reached within the horizon, returns T+1.

        Parameters
        ----------
        lower_bounds : np.ndarray, shape (m, T)
            L_t(π_i; α/m) for each policy i and time t.
        upper_bounds : np.ndarray, shape (m, T)
            U_t(π_i; α/m) for each policy i and time t.

        Returns
        -------
        int
            1-indexed stopping time τ, or T+1 if the stopping condition is
            never met within the horizon.
        """
        if lower_bounds.ndim != 2 or upper_bounds.ndim != 2:
            raise ValueError("lower_bounds and upper_bounds must be 2D arrays.")
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError(
                f"Shape mismatch: lower_bounds {lower_bounds.shape} vs "
                f"upper_bounds {upper_bounds.shape}."
            )

        m, T = lower_bounds.shape

        # Trivial case: only one policy, no competitor to beat.
        if m == 1:
            return T + 1

        # "Second-max trick": for each policy i and time t, compute
        #   max_{j ≠ i} U_t(π_j)
        # by sorting upper bounds and using the top-two values.
        sorted_U = np.sort(upper_bounds, axis=0)   # (m, T), ascending
        max1 = sorted_U[-1, :]                      # (T,) — global max
        max2 = sorted_U[-2, :]                      # (T,) — second max

        # For each (i, t): if π_i achieves the global max, the best competitor
        # is max2; otherwise it is max1.
        is_argmax = upper_bounds == max1[np.newaxis, :]   # (m, T)
        max_excl = np.where(is_argmax, max2[np.newaxis, :], max1[np.newaxis, :])

        # Stopping condition: ∃i s.t. L_t(π_i) > max_{j≠i} U_t(π_j)
        any_stopped = (lower_bounds > max_excl).any(axis=0)   # (T,)

        first_stop = np.where(any_stopped)[0]
        return int(first_stop[0]) + 1 if len(first_stop) > 0 else T + 1
