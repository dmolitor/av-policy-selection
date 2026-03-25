"""
Unit tests for PolicySelector.

All tests use hand-constructed (m, T) arrays with exact known outputs.
"""

import numpy as np
import pytest

from av_policy_selection import PolicySelector


# ── optimal_set tests ──────────────────────────────────────────────────────────

def test_optimal_set_dominant():
    """Both policies remain in S_t when each upper bound ≥ max lower bound."""
    # m=2, T=3
    # Policy 0: L=0.5, U=0.9  |  Policy 1: L=0.3, U=0.8
    # max_lower = 0.5 at all t
    # U[0]=0.9 ≥ 0.5 → True; U[1]=0.8 ≥ 0.5 → True
    lower = np.array([[0.5, 0.5, 0.5],
                       [0.3, 0.3, 0.3]])
    upper = np.array([[0.9, 0.9, 0.9],
                       [0.8, 0.8, 0.8]])
    result = PolicySelector.optimal_set(lower, upper)
    assert result.shape == (2, 3)
    assert result.all(), "Both policies should be in S_t at all t."


def test_optimal_set_eliminated():
    """Policy 1 is eliminated when its upper bound < max lower bound."""
    # m=2, T=3
    # Policy 0: L=0.6, U=0.95  |  Policy 1: L=0.1, U=0.4
    # max_lower = 0.6; U[1]=0.4 < 0.6 → False
    lower = np.array([[0.6, 0.6, 0.6],
                       [0.1, 0.1, 0.1]])
    upper = np.array([[0.95, 0.95, 0.95],
                       [0.40, 0.40, 0.40]])
    result = PolicySelector.optimal_set(lower, upper)
    assert result[0].all(), "Policy 0 should always be in S_t."
    assert not result[1].any(), "Policy 1 should never be in S_t."


def test_optimal_set_all_equal():
    """All policies remain in S_t when CSs are identical."""
    # m=3, T=4; L=0.3, U=0.7 for all
    lower = np.full((3, 4), 0.3)
    upper = np.full((3, 4), 0.7)
    result = PolicySelector.optimal_set(lower, upper)
    assert result.shape == (3, 4)
    assert result.all(), "All policies should be in S_t when CSs are equal."


def test_optimal_set_shape_mismatch_raises():
    """Mismatched shapes should raise ValueError."""
    lower = np.zeros((2, 5))
    upper = np.zeros((3, 5))
    with pytest.raises(ValueError, match="Shape mismatch"):
        PolicySelector.optimal_set(lower, upper)


def test_optimal_set_not_2d_raises():
    """1D arrays should raise ValueError."""
    lower = np.zeros(5)
    upper = np.zeros(5)
    with pytest.raises(ValueError, match="2D"):
        PolicySelector.optimal_set(lower, upper)


# ── stopping_time tests ────────────────────────────────────────────────────────

def test_stopping_time_known_tau():
    """τ=3 when policy 0 first dominates policy 1 at t=3 (1-indexed)."""
    # m=2, T=5
    # t=1: L[0]=0.4, U[1]=0.6 → not stopped
    # t=2: L[0]=0.7, U[1]=0.75 → not stopped (0.7 < 0.75)
    # t=3: L[0]=0.9, U[1]=0.7  → stopped (0.9 > 0.7)
    lower = np.array([
        [0.4, 0.7, 0.9, 0.9, 0.9],   # policy 0
        [0.1, 0.1, 0.1, 0.1, 0.1],   # policy 1
    ])
    upper = np.array([
        [0.95, 0.95, 0.95, 0.95, 0.95],  # policy 0
        [0.60, 0.75, 0.70, 0.70, 0.70],  # policy 1
    ])
    tau = PolicySelector.stopping_time(lower, upper)
    assert tau == 3, f"Expected τ=3, got {tau}."


def test_stopping_time_never_stops():
    """τ=T+1 when CSs overlap throughout the horizon."""
    # m=2, T=4; L[0]=0.4, U[1]=0.6 always — never stopped
    lower = np.array([[0.4, 0.4, 0.4, 0.4],
                       [0.2, 0.2, 0.2, 0.2]])
    upper = np.array([[0.8, 0.8, 0.8, 0.8],
                       [0.6, 0.6, 0.6, 0.6]])
    tau = PolicySelector.stopping_time(lower, upper)
    assert tau == 5, f"Expected τ=T+1=5, got {tau}."


def test_stopping_time_three_policies():
    """τ=4 when π_0 first dominates all others at t=4."""
    # m=3, T=6
    # policy 0 lower rises to 0.85 at t=4; competitors' upper = 0.7 throughout
    lower = np.array([
        [0.1, 0.5, 0.7, 0.85, 0.85, 0.85],   # policy 0
        [0.1, 0.1, 0.1, 0.10, 0.10, 0.10],   # policy 1
        [0.1, 0.1, 0.1, 0.10, 0.10, 0.10],   # policy 2
    ])
    upper = np.array([
        [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],  # policy 0
        [0.90, 0.90, 0.90, 0.70, 0.70, 0.70],  # policy 1
        [0.80, 0.80, 0.80, 0.70, 0.70, 0.70],  # policy 2
    ])
    tau = PolicySelector.stopping_time(lower, upper)
    assert tau == 4, f"Expected τ=4, got {tau}."


def test_stopping_time_m1():
    """τ=T+1 for m=1 (trivial: no competitors)."""
    lower = np.array([[0.1, 0.5, 0.9, 0.95]])
    upper = np.array([[0.8, 0.9, 0.99, 0.999]])
    tau = PolicySelector.stopping_time(lower, upper)
    assert tau == 5, f"Expected τ=T+1=5, got {tau}."


def test_stopping_time_shape_mismatch_raises():
    """Mismatched shapes should raise ValueError."""
    lower = np.zeros((2, 5))
    upper = np.zeros((2, 6))
    with pytest.raises(ValueError, match="Shape mismatch"):
        PolicySelector.stopping_time(lower, upper)
