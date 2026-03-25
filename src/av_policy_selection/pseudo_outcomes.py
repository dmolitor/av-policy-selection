"""
Importance weights and doubly-robust pseudo-outcomes for off-policy inference.

These correspond to definitions and equations in:
  Luedtke & Soni (2024), "Anytime-Valid Off-Policy Inference for Contextual Bandits"
  [eq. predictable-importance-weights, dr-outcomes-lower, dr-outcomes-upper]
"""

import numpy as np


def importance_weights(pi_probs: np.ndarray, h_probs: np.ndarray) -> np.ndarray:
    """Compute importance weights w_t = π(A_t|X_t) / h_t(A_t|X_t).

    Parameters
    ----------
    pi_probs : np.ndarray, shape (T,)
        Target policy probabilities π(A_t|X_t) at the observed actions.
    h_probs : np.ndarray, shape (T,)
        Logging policy probabilities h_t(A_t|X_t) at the observed actions.

    Returns
    -------
    np.ndarray, shape (T,)
        Importance weights w_t. [eq. predictable-importance-weights]
    """
    pi_probs = np.asarray(pi_probs, dtype=float)
    h_probs = np.asarray(h_probs, dtype=float)
    return pi_probs / h_probs


def iw_pseudo_outcomes(
    rewards: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute importance-weighted pseudo-outcomes (k=0 special case).

    φ_IWL_t = w_t * R_t
    φ_IWU_t = w_t * (1 - R_t)

    Parameters
    ----------
    rewards : np.ndarray, shape (T,)
        Observed rewards R_t ∈ [0, 1].
    weights : np.ndarray, shape (T,)
        Importance weights w_t.

    Returns
    -------
    phi_lower : np.ndarray, shape (T,)
        Lower pseudo-outcomes φ_IWL.
    phi_upper : np.ndarray, shape (T,)
        Upper pseudo-outcomes φ_IWU.
    """
    rewards = np.asarray(rewards, dtype=float)
    weights = np.asarray(weights, dtype=float)
    phi_lower = weights * rewards
    phi_upper = weights * (1.0 - rewards)
    return phi_lower, phi_upper


def dr_pseudo_outcomes(
    rewards: np.ndarray,
    weights: np.ndarray,
    r_hat: np.ndarray,
    r_hat_pi_mean_lower: np.ndarray,
    r_hat_pi_mean_upper: np.ndarray,
    k: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute doubly-robust pseudo-outcomes.

    φ_DRL_t = w_t*(R_t - min(r̂_t, k_t/w_t)) + E_{a~π}[min(r̂_t(X_t,a), k_t/w_t)]
    φ_DRU_t = w_t*((1-R_t) - min(1-r̂_t, k_t/w_t)) + E_{a~π}[min(1-r̂_t(X_t,a), k_t/w_t)]

    Note: φ_DRL_t >= -k_t always (mathematical invariant).
    [eq. dr-outcomes-lower, dr-outcomes-upper]

    Parameters
    ----------
    rewards : np.ndarray, shape (T,)
        Observed rewards R_t ∈ [0, 1].
    weights : np.ndarray, shape (T,)
        Importance weights w_t.
    r_hat : np.ndarray, shape (T,)
        Reward model predictions r̂_t(X_t, A_t) at observed contexts/actions.
    r_hat_pi_mean_lower : np.ndarray, shape (T,)
        E_{a~π}[min(r̂_t(X_t, a), k_t/w_t)] — truncated expected reward under
        target policy for the lower bound. For binary actions {0,1} with π(1|x)=p:
          r_hat_pi_mean_lower[t] = p*min(r̂_1, k/w_t) + (1-p)*min(r̂_0, k/w_t)
    r_hat_pi_mean_upper : np.ndarray, shape (T,)
        E_{a~π}[min(1-r̂_t(X_t, a), k_t/w_t)] — truncated expected complement
        reward under target policy for the upper bound. For binary actions:
          r_hat_pi_mean_upper[t] = p*min(1-r̂_1, k/w_t) + (1-p)*min(1-r̂_0, k/w_t)
    k : float or np.ndarray
        Truncation parameter k_t >= 0.

    Returns
    -------
    phi_lower : np.ndarray, shape (T,)
        Lower DR pseudo-outcomes φ_DRL. Satisfies φ_DRL >= -k always.
    phi_upper : np.ndarray, shape (T,)
        Upper DR pseudo-outcomes φ_DRU.
    """
    rewards = np.asarray(rewards, dtype=float)
    weights = np.asarray(weights, dtype=float)
    r_hat = np.asarray(r_hat, dtype=float)
    r_hat_pi_mean_lower = np.asarray(r_hat_pi_mean_lower, dtype=float)
    r_hat_pi_mean_upper = np.asarray(r_hat_pi_mean_upper, dtype=float)
    k = np.asarray(k, dtype=float)

    # Safe division: when w_t = 0 the entire importance-weighted term is 0,
    # so the threshold value is irrelevant — use k as a harmless fallback.
    safe_weights = np.where(weights > 0, weights, 1.0)
    threshold = k / safe_weights  # k_t / w_t

    # Lower: φ_DRL = w_t*(R_t - min(r̂_t, k_t/w_t)) + E_{a~π}[min(r̂_t(X_t,a), k_t/w_t)]
    phi_lower = (
        weights * (rewards - np.minimum(r_hat, threshold)) + r_hat_pi_mean_lower
    )

    # Upper: φ_DRU = w_t*((1-R_t) - min(1-r̂_t, k_t/w_t)) + E_{a~π}[min(1-r̂_t(X_t,a), k_t/w_t)]
    phi_upper = (
        weights * ((1.0 - rewards) - np.minimum(1.0 - r_hat, threshold))
        + r_hat_pi_mean_upper
    )

    return phi_lower, phi_upper
