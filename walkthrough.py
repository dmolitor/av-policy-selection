#!/usr/bin/env python
"""
Anytime-Valid Optimal Policy Identification — Walkthrough
==========================================================

Demonstrates the optimal policy identification methods from:
    resources/anytime-valid-optimal-policy-identification.tex

Eight demonstrations:
  Demo 1  — Stopping time τ vs. (Δ_min, M) for LIL and Betting CSs
  Demo 1b — Identification probability P(τ ≤ t) vs. sample size (B1)
  Demo 2A — Single-run S_t heatmap + per-policy CS ribbon plots
  Demo 2A-E1 — Confidence sequence width trajectory (E1)
  Demo 2B — Aggregate coverage P(π* ∈ S_t) across trials (LIL, Hoeffding, Betting)
  Demo 2C — Stopping time τ vs. logging-policy overlap (4 severity × 3 bad-fraction levels)
  Demo 3  — Stopping time τ vs. policy-class size M (Bonferroni scaling, B2)
  Demo 4  — Nuisance model sensitivity: effect of reward predictor quality (C1)

Usage
-----
Test mode (fast, small parameters — verifies the code runs end-to-end):
    python walkthrough.py

Full mode (production parameters — slow, intended for offline runs):
    FULL_RUN=1 python walkthrough.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from plotnine import (
    aes,
    coord_cartesian,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_hline,
    geom_line,
    geom_ribbon,
    geom_tile,
    geom_violin,
    ggplot,
    labs,
    labeller,
    scale_color_manual,
    scale_fill_manual,
    scale_x_log10,
    scale_y_continuous,
    stat_summary,
    theme,
    theme_bw,
    theme_minimal,
)

from av_policy_selection import (
    BanditSimulator,
    BettingConfidenceSequence,
    HoeffdingConfidenceBound,
    LILConfidenceSequence,
    OLSRewardPredictor,
    PolicySelector,
    RewardPredictor,
)

# ═══════════════════════════════════════════════════════════════════════════════
# MODE: set FULL_RUN=1 in environment for production parameters
# ═══════════════════════════════════════════════════════════════════════════════
FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"

# ── Full / test parameter sets ─────────────────────────────────────────────────
if FULL_RUN:
    # ── Demo 1 parameters ─────────────────────────────────────────────────────
    # Values of Δ_min to sweep. Each entry defines a suboptimal policy whose
    # true value falls exactly Δ_min below the optimal (see eq_gap in the DGP
    # section). Must be strictly less than GAP=0.35; entries ≥ GAP are skipped.
    DELTA_MIN_GRID   = [
        # 0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25
    ]

    # Policy-set sizes to sweep. For each m, the set contains 1 optimal policy,
    # (m-1) identical suboptimal policies at the current Δ_min, PLUS the logging
    # policy (always included). Total policy count is therefore m+1, and the
    # α/(m+1) Bonferroni correction is applied. Larger m widens each per-policy
    # CS, making stopping harder — consistent with the log(m) term in Theorem 1.
    M_GRID           = [2, 5, 10, 20]

    # Number of independent trials used to estimate E[τ] for the LIL CS method.
    # LIL bounds are O(T) to compute, so this can be large.
    N_TRIALS_LIL     = 10

    # Number of independent trials for the Betting CS method. Kept smaller
    # because BettingConfidenceSequence.bounds() is O(T²) per call (brentq
    # root-finding at every t), making it ~T× slower than LIL per trial.
    N_TRIALS_BETTING = 10

    # Hard horizon cap. If τ > T_MAX the trial is censored at T_MAX. Results
    # near T_MAX indicate the horizon is too short to resolve stopping for that
    # (Δ_min, m) combination and should be interpreted as lower bounds on E[τ].
    T_MAX            = 10000


    # ── Demo 2A parameters ────────────────────────────────────────────────────
    # Number of time steps in the single illustrative run. Long enough that
    # suboptimal policies (Δ ≥ 0.10) are visibly eliminated from S_t.
    T_SINGLE         = 10000

    # Subsampling stride for the heatmap x-axis. Every T_STRIDE-th time step
    # is plotted; reduces the number of tiles without losing visual resolution.
    T_STRIDE         = 20

    # ── Demo 2B parameters ────────────────────────────────────────────────────
    # Number of independent trials over which P(π* ∈ S_t) is estimated.
    # More trials → lower Monte Carlo error in the coverage estimate.
    N_COV            = 200

    # Horizon for each coverage trial. Should be long enough that the marginal
    # coverage curve has stabilised (i.e. CSs have shrunk enough to reliably
    # contain π*).
    T_COV            = 2000

    # ── Demo 2C parameters ────────────────────────────────────────────────────
    # Partial-overlap DGP: the problematic subpopulation (X < frac_bad) has
    # degraded logging-policy overlap controlled by p_bad = P(A=1|X); the rest
    # use balanced logging (p=0.5, w≤2).
    # Four severity levels (max w in the problematic subpopulation):
    #   Mild     (p_bad=1/3)   → max w ≈ 3
    #   Moderate (p_bad=0.1)   → max w ≈ 10
    #   High     (p_bad=0.01)  → max w ≈ 100
    #   Severe   (p_bad=0.001) → max w ≈ 1000
    # Three bad-subpopulation sizes: 5%, 10%, 20% of contexts.
    P_BAD_GRID       = [1.0/3, 0.1, 0.01, 0.001]  # → max w ≈ 3, 10, 100, 1000
    FRAC_BAD_GRID    = [0.05, 0.10, 0.20]          # fraction of contexts in bad subpop
    DELTA_MIN_OV     = 0.10              # fixed gap between π* and suboptimal policies
    M_OV             = 3                 # 1 optimal + (M_OV-1) suboptimal policies
    N_TRIALS_OV      = 100              # trials per (p_bad, frac_bad) combination
    T_MAX_OV         = 50000            # separate horizon cap for Demo 2C

    # ── Parallelism ───────────────────────────────────────────────────────────
    # Number of joblib worker threads. -1 means use all available CPU cores.
    N_JOBS           = -1

    # ── Demo 3 parameters (B2: τ vs M scaling) ────────────────────────────────
    # Policy-set sizes to sweep with fixed gap DELTA_B2. Wider range than
    # Demo 1's M_GRID to expose the log(M) Bonferroni cost over decades.
    M_SWEEP      = [2, 5, 10, 20, 50, 100]

    # Fixed suboptimality gap for Demo 3's M sweep.
    DELTA_B2     = 0.10

    # Number of independent trials per M value in Demo 3. LIL is O(T) per
    # trial so this can be generous.
    N_TRIALS_B2  = 50

    # ── Demo 4 parameters (C1: nuisance sensitivity) ──────────────────────────
    # Fixed suboptimality gap for Demo 4's predictor comparison.
    DELTA_C1     = 0.15

    # Trials per predictor type in Demo 4.
    N_TRIALS_C1  = 100
else:
    # Minimal parameters — smoke-test only; verifies correctness, not outputs.
    DELTA_MIN_GRID   = [0.25]          # one gap value
    M_GRID           = [2]             # one policy-set size
    N_TRIALS_LIL     = 3
    N_TRIALS_BETTING = 2
    T_MAX            = 50              # Betting stays O(50²) per trial
    T_SINGLE         = 100             # Demo 2A single-run length
    T_STRIDE         = 10
    N_COV            = 3               # Demo 2B coverage trials
    T_COV            = 50              # Betting O(50²) per coverage trial
    N_JOBS           = 2
    P_BAD_GRID       = [1.0/3, 0.1, 0.01, 0.001]
    FRAC_BAD_GRID    = [0.10]          # one bad-fraction level
    DELTA_MIN_OV     = 0.10
    M_OV             = 3
    N_TRIALS_OV      = 2
    T_MAX_OV         = 50             # keep test fast; O(50²) per Betting call
    M_SWEEP      = [2, 5]
    DELTA_B2     = 0.10
    N_TRIALS_B2  = 2
    DELTA_C1     = 0.15
    N_TRIALS_C1  = 2

# Overall significance level α. Per-policy CSs are built at α/m (Bonferroni),
# so the simultaneous guarantee P(ν(π) ∈ CS_t(π) ∀t, ∀π) ≥ 1-α holds exactly.
ALPHA = 0.10

# ═══════════════════════════════════════════════════════════════════════════════
# DGP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# X = (X1, X2, X3) ~ Uniform(0, 1)^3
# E[R | X, A=a] = BETA[a, 0] + BETA[a, 1]*X1 + BETA[a, 2]*X2 + BETA[a, 3]*X3
# R ~ Bernoulli(E[R|X,A])
#
# Analytically:
#   EV[0] = 0.25 + 0.5*(0.10+0.10+0.10) = 0.40
#   EV[1] = 0.55 + 0.5*(0.10+0.10+0.10) = 0.70   (gap = 0.30)
#
# ν(π_p) = p * EV[1] + (1-p) * EV[0]
# p_sub(Δ_min) = (0.30 - Δ_min) / 0.30  (valid for Δ_min < 0.30)

# Reward-model coefficient matrix. Row a gives [intercept, β_1, β_2, β_3] for
# action a, so E[R|X,A=a] = BETA[a,0] + BETA[a,1]*X1 + BETA[a,2]*X2 + BETA[a,3]*X3.
# The intercepts (0.25 vs 0.55) create a large baseline gap between actions;
# the shared slope coefficients (0.10, 0.10, 0.10) mean covariates shift both
# actions equally, so the optimal action does not depend on context.
BETA = np.array([
    [0.25, 0.10, 0.10, 0.10],   # A=0: E[R|A=0] = 0.25 + 0.10*(X1+X2+X3), mean = 0.40
    [0.55, 0.10, 0.10, 0.10],   # A=1: E[R|A=1] = 0.55 + 0.10*(X1+X2+X3), mean = 0.70
])

# DR truncation parameter k ≥ 0. Controls the variance–bias trade-off of the
# doubly-robust pseudo-outcomes: φ_DRL ∈ [-k, 1+k]. k=0 recovers plain
# importance weighting (no reward-model correction); k>0 exploits the fitted
# reward predictor to reduce variance. With the uniform logging policy
# (LOGGING_P=0.5), importance weights are at most 2, so k≥2 effectively
# removes all truncation bias while still enabling strong DR variance reduction.
K = 1.0

# Global random seed for reproducibility. Each parallel worker derives its own
# independent seed from this base value, so results are deterministic regardless
# of the number of threads used.
SEED  = 969808

# Analytically derived expected reward under each pure action, averaged over
# X ~ Uniform(0,1)^3: EV[a] = BETA[a,0] + 0.5*(BETA[a,1]+BETA[a,2]+BETA[a,3]).
# EV[0] = 0.325, EV[1] = 0.675.
EV  = BETA[:, 0] + 0.5 * BETA[:, 1:].sum(axis=1)

# True value gap between the optimal policy (always A=1) and any policy π_p:
# Δ(π_p) = EV[1] - ν(π_p) = 0.35*(1-p). This is the Δ_min referenced in
# Theorem 1 of the paper when all suboptimal policies share the same gap.
GAP = float(EV[1] - EV[0])   # = 0.35


def true_value(p: float) -> float:
    return p * EV[1] + (1.0 - p) * EV[0]


def mu(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    b = BETA[A.astype(int)]
    return b[:, 0] + (X * b[:, 1:]).sum(axis=1)


def make_reward_fn(rng: np.random.Generator):
    def reward_fn(ctx, act):
        return rng.binomial(1, mu(ctx, act)).astype(float)
    return reward_fn


def make_context_fn(rng: np.random.Generator):
    def context_fn(T):
        return rng.uniform(0, 1, (T, 3))
    return context_fn


# Probability of choosing A=1 under the logging (data-collection) policy.
# 0.50 = uniform randomisation, giving importance weights w_t = π(A_t)/0.5 ∈ {0, 2},
# which keeps weights bounded and satisfies the overlap assumption of the paper.
LOGGING_P = 0.50


def logging_policy(ctx: np.ndarray, act: np.ndarray) -> np.ndarray:
    return np.where(act == 1, LOGGING_P, 1.0 - LOGGING_P).astype(float)


def make_policy(p: float):
    def policy(ctx, act):
        return np.where(act == 1, p, 1.0 - p).astype(float)
    return policy


def make_sim(rng: np.random.Generator) -> BanditSimulator:
    """Build a fresh BanditSimulator with its own RNG (safe for parallel use)."""
    return BanditSimulator(
        logging_policy=logging_policy,
        target_policy=logging_policy,
        reward_fn=make_reward_fn(rng),
        context_fn=make_context_fn(rng),
        actions=np.array([0, 1]),
        true_policy_value=true_value(LOGGING_P),
        k=K,
        rng=rng,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Local reward predictor classes for Demo 4 (nuisance sensitivity)
# ═══════════════════════════════════════════════════════════════════════════════

class _OracleRewardPredictor(RewardPredictor):
    """Uses the true μ function directly — zero bias by construction."""

    def __init__(self, actions: np.ndarray):
        self.actions = np.asarray(actions)
        self._fitted = True

    def fit(self, contexts, actions, rewards):
        return self   # no-op: oracle needs no fitting

    def predict(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.clip(mu(contexts, np.asarray(actions)), 0.0, 1.0)


class _ConstantRewardPredictor(RewardPredictor):
    """Always predicts the global mean reward — ignores covariates and action."""

    def __init__(self, actions: np.ndarray):
        self.actions = np.asarray(actions)
        self._fitted = False
        self._mean = 0.0

    def fit(self, contexts, actions, rewards):
        self._mean = float(np.mean(rewards))
        self._fitted = True
        return self

    def predict(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(actions)), self._mean)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker functions (module-level so joblib can pickle them)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_trial_lil(seed: int, T: int, policies: list, alpha_policy: float) -> int:
    """Single LIL trial; returns min(τ, T)."""
    rng = np.random.default_rng(seed)
    sim = make_sim(rng)
    data = sim.simulate(T)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(data.contexts, data.actions, data.rewards)
    all_L, all_U = [], []
    for policy_fn in policies:
        phi_l, phi_u = sim.evaluate_policy(
            data, policy_fn, kind="dr", reward_predictor=predictor
        )
        L, U = LILConfidenceSequence(alpha=alpha_policy, k=K).bounds(phi_l, phi_u)
        all_L.append(L)
        all_U.append(U)
    tau = PolicySelector.stopping_time(np.stack(all_L), np.stack(all_U))
    return min(tau, T)


def _run_trial_betting(seed: int, T: int, policies: list, alpha_policy: float) -> int:
    """Single Betting trial; returns min(τ, T)."""
    rng = np.random.default_rng(seed)
    sim = make_sim(rng)
    data = sim.simulate(T)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(data.contexts, data.actions, data.rewards)
    all_L, all_U = [], []
    for policy_fn in policies:
        phi_l, phi_u = sim.evaluate_policy(
            data, policy_fn, kind="dr", reward_predictor=predictor
        )
        L, U = BettingConfidenceSequence(alpha=alpha_policy, k=K).bounds(phi_l, phi_u)
        all_L.append(L)
        all_U.append(U)
    tau = PolicySelector.stopping_time(np.stack(all_L), np.stack(all_U))
    return min(tau, T)


def _run_trial_predictor(
    seed: int, T: int, policies: list, alpha_policy: float, predictor_type: str,
) -> int:
    """Single LIL trial with a specified reward predictor; returns min(τ, T).

    Parameters
    ----------
    predictor_type : str
        ``"oracle"`` uses the true μ function directly (zero model bias).
        ``"ols"`` uses ordinary least-squares linear regression (current default).
        ``"constant"`` always predicts the global mean reward (ignores covariates).
    """
    rng = np.random.default_rng(seed)
    sim = make_sim(rng)
    data = sim.simulate(T)

    if predictor_type == "oracle":
        predictor = _OracleRewardPredictor(actions=np.array([0, 1]))
    elif predictor_type == "constant":
        predictor = _ConstantRewardPredictor(actions=np.array([0, 1]))
    else:  # "ols"
        predictor = OLSRewardPredictor(actions=np.array([0, 1]))

    predictor.fit(data.contexts, data.actions, data.rewards)
    all_L, all_U = [], []
    for policy_fn in policies:
        phi_l, phi_u = sim.evaluate_policy(
            data, policy_fn, kind="dr", reward_predictor=predictor
        )
        L, U = LILConfidenceSequence(alpha=alpha_policy, k=K).bounds(phi_l, phi_u)
        all_L.append(L)
        all_U.append(U)
    tau = PolicySelector.stopping_time(np.stack(all_L), np.stack(all_U))
    return min(tau, T)


def _run_trial_overlap(
    seed: int, T: int, policies: list, alpha_policy: float, p_bad: float,
    frac_bad: float = 0.10,
) -> int:
    """Single Betting trial with partial-overlap DGP; returns min(τ, T).

    DGP: X ~ Uniform(0, 1),  R ~ Bernoulli(0.25 + 0.30·X + 0.30·A).
    ν(π*) = E[0.25 + 0.30·X + 0.30] = 0.70;  gap_OV = 0.30.

    Logging policy: P(A=1|X) = 0.5    for X ≥ frac_bad  (normal subpop, w ≤ 2)
                    P(A=1|X) = p_bad   for X < frac_bad  (problematic subpop)

    Parameters
    ----------
    p_bad : float
        P(A=1|X) in the problematic subpopulation.
        p_bad = 1/3   → max w ≈ 3    (mild)
        p_bad = 0.1   → max w ≈ 10   (moderate)
        p_bad = 0.01  → max w ≈ 100  (high)
        p_bad = 0.001 → max w ≈ 1000 (severe)
    frac_bad : float
        Fraction of the context space in the problematic subpopulation,
        i.e. P(X < frac_bad). E.g. 0.05, 0.10, or 0.20.
    """
    rng = np.random.default_rng(seed)

    def ov_context_fn(T_):
        return rng.uniform(0, 1, (T_, 1))   # 1-D context, shape (T, 1)

    def ov_reward_fn(ctx, act):
        x = ctx[:, 0]
        prob = 0.25 + 0.30 * x + 0.30 * act
        return rng.binomial(1, prob).astype(float)

    def ov_log_policy(ctx, act):
        x = ctx[:, 0]
        in_bad = x < frac_bad
        p = np.where(in_bad, np.where(act == 1, p_bad, 1.0 - p_bad), 0.5)
        return p.astype(float)

    sim = BanditSimulator(
        logging_policy=ov_log_policy,
        target_policy=ov_log_policy,
        reward_fn=ov_reward_fn,
        context_fn=ov_context_fn,
        actions=np.array([0, 1]),
        true_policy_value=0.70,
        k=K,
        rng=rng,
    )
    data = sim.simulate(T)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(data.contexts, data.actions, data.rewards)
    all_L, all_U = [], []
    for policy_fn in policies:
        phi_l, phi_u = sim.evaluate_policy(
            data, policy_fn, kind="dr", reward_predictor=predictor
        )
        L, U = BettingConfidenceSequence(alpha=alpha_policy, k=K).bounds(phi_l, phi_u)
        all_L.append(L)
        all_U.append(U)
    tau = PolicySelector.stopping_time(np.stack(all_L), np.stack(all_U))
    return min(tau, T)


def _run_coverage_trial(
    seed: int, T: int, probs: list, alpha_policy: float, method: str = "lil",
    correction: str = "bonferroni",
) -> np.ndarray:
    """Single coverage trial; returns bool array (T,) — True iff π* ∈ S_t.

    Parameters
    ----------
    method : str
        ``"lil"`` for LILConfidenceSequence, ``"betting"`` for BettingConfidenceSequence.
    correction : str
        ``"bonferroni"`` uses alpha_policy as-is (plain Bonferroni).
        ``"effective_m"`` adjusts alpha upward using the average pairwise
        correlation of pseudo-outcomes across policies.  With M tests sharing
        positively correlated data, the effective number of independent tests
        M_eff = 1 + (M-1)*(1-ρ̄) < M, so alpha_each = ALPHA/M_eff > alpha_policy.
        Revert to Bonferroni by passing correction="bonferroni".
    """
    if method == "lil":
        cs_cls = LILConfidenceSequence
    elif method == "hoeffding":
        cs_cls = HoeffdingConfidenceBound
    else:
        cs_cls = BettingConfidenceSequence
    rng = np.random.default_rng(seed)
    sim = make_sim(rng)
    data = sim.simulate(T)
    predictor = OLSRewardPredictor(actions=np.array([0, 1]))
    predictor.fit(data.contexts, data.actions, data.rewards)

    all_phi_l, all_phi_u = [], []
    for p in probs:
        phi_l, phi_u = sim.evaluate_policy(
            data, make_policy(p), kind="dr", reward_predictor=predictor
        )
        all_phi_l.append(phi_l)
        all_phi_u.append(phi_u)

    if correction == "effective_m":
        M = len(probs)
        corr_mat = np.corrcoef(np.stack(all_phi_l))   # (M, M)
        rho_bar = (corr_mat.sum() - M) / (M * (M - 1))
        M_eff = 1.0 + (M - 1) * (1.0 - rho_bar)
        alpha_each = ALPHA / M_eff
    else:  # "bonferroni"
        alpha_each = alpha_policy

    all_L, all_U = [], []
    for phi_l, phi_u in zip(all_phi_l, all_phi_u):
        L, U = cs_cls(alpha=alpha_each, k=K).bounds(phi_l, phi_u)
        all_L.append(L)
        all_U.append(U)
    in_set = PolicySelector.optimal_set(np.stack(all_L), np.stack(all_U))
    return in_set[0]   # π* is index 0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Anytime-Valid Optimal Policy Identification — Walkthrough")
print(f"  Mode: {'FULL' if FULL_RUN else 'TEST (set FULL_RUN=1 for production)'}")
print("=" * 70)
print(f"\nDGP:  EV[0]={EV[0]:.3f},  EV[1]={EV[1]:.3f},  gap={GAP:.3f}")
print(f"      K={K},  α={ALPHA}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 1: Stopping time τ vs. Δ_min × M
# ═══════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  Demo 1: Stopping time τ vs. Δ_min × M")
print("─" * 70)

# Base seed stream for Demo 1 (deterministic, independent per cell/trial)
_d1_seed = np.random.SeedSequence(SEED)

demo1_rows = []

# Build the full list of (m, delta_min) cells upfront so tqdm can show a
# meaningful total count across the outer grid.
_d1_cells = [
    (m, delta_min)
    for m in M_GRID
    for delta_min in DELTA_MIN_GRID
    if (GAP - delta_min) / GAP >= 0
]
_d1_methods = [
    ("LIL CS",     _run_trial_lil,     N_TRIALS_LIL),
    ("Betting CS", _run_trial_betting, N_TRIALS_BETTING),
]
_d1_total = len(_d1_cells) * len(_d1_methods)

with tqdm(total=_d1_total, desc="Demo 1 grid", unit="cell") as pbar:
    for m, delta_min in _d1_cells:
        p_sub = (GAP - delta_min) / GAP
        # Logging policy is always included; total policy count = m + 1.
        policies = [make_policy(1.0)] + [make_policy(p_sub)] * (m - 1) + [make_policy(LOGGING_P)]
        alpha_policy = ALPHA / (m + 1)

        # Unique seed block per (m, delta_min)
        cell_key = m * 1000 + int(round(delta_min * 1000))

        for method, worker, n_t in _d1_methods:
            seeds = [cell_key * 10000 + i for i in range(n_t)]
            pbar.set_postfix(M=m, delta=f"{delta_min:.2f}", method=method)
            taus = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(worker)(s, T_MAX, policies, alpha_policy)
                for s in tqdm(seeds, desc=f"  trials", leave=False, unit="trial")
            )
            taus_arr = np.array(taus)
            for tau in taus:
                demo1_rows.append({
                    "M": m, "delta_min": delta_min, "method": method, "tau": tau,
                })
            tqdm.write(
                f"  M={m:2d}  Δ_min={delta_min:.2f}  {method:<8}"
                f"  mean τ={np.mean(taus_arr):7.1f}  median τ={np.median(taus_arr):6.0f}"
                f"  P(censored)={np.mean(taus_arr == T_MAX):.1%}"
            )
            pbar.update(1)

df_stop = pd.DataFrame(demo1_rows)
# M as an ordered string factor so the legend reads "M=2, M=5, ..." in order.
_m_levels = [f"M={m}" for m in sorted(df_stop["M"].unique())]
df_stop["M_label"] = pd.Categorical(
    "M=" + df_stop["M"].astype(str), categories=_m_levels, ordered=True
)
_m_colors = dict(zip(_m_levels, ["#1a9641", "#4393c3", "#d6604d", "#984ea3"]))

# ── Summary function — swap to np.median, np.min, etc. without re-running ─────
_d1_summary_fn = np.mean

plot_stop = (
    ggplot(df_stop, aes(x="delta_min", y="tau", color="M_label", group="M_label"))
    + stat_summary(fun_y=_d1_summary_fn, geom="line")
    + facet_wrap("method")
    + scale_color_manual(values=_m_colors)
    + scale_y_continuous(breaks=[0, 1000, 2000, 3000, 4000, 5000, 6000])
    + labs(
        x="Suboptimality gap size",
        y="Mean stopping time (sample size)",
        color="Policy-class size",
        title="Stopping time for optimal policy identification",
        subtitle=f"α={ALPHA}",
    )
    + theme_minimal()
    + theme(figure_size=(12, 5))
)
plot_stop.save(
    "./figures/stopping-time.png",
    dpi=300,
    height=3,
    width=6
)

# ── Demo 1b (B1): Identification probability P(τ ≤ t) vs sample size ──────────
# Reuses Demo 1's raw τ data — no additional simulation required.
# Fix M=2 (clearest signal) and show ECDF curves per Δ_min for LIL CS.

_b1_M = M_GRID[0]   # smallest M; most interpretable
_b1_data = df_stop[(df_stop["M"] == _b1_M) & (df_stop["method"] == "LIL CS")]
_b1_delta_vals = sorted(_b1_data["delta_min"].unique())
_b1_delta_labels = [f"Δ={d:.2f}" for d in _b1_delta_vals]
_b1_delta_order  = _b1_delta_labels

_b1_palette = ["#1a9641", "#4393c3", "#d6604d", "#984ea3", "#ff7f00"]
_b1_colors = dict(zip(_b1_delta_labels, _b1_palette[:len(_b1_delta_labels)]))

_ecdf_stride = max(1, T_MAX // 300)
_ecdf_rows = []
for d, d_label in zip(_b1_delta_vals, _b1_delta_labels):
    taus_d = _b1_data[_b1_data["delta_min"] == d]["tau"].values
    for t in range(1, T_MAX + 1, _ecdf_stride):
        _ecdf_rows.append({
            "t": t,
            "p_identified": float(np.mean(taus_d <= t)),
            "delta_label": d_label,
        })

df_b1 = pd.DataFrame(_ecdf_rows)
df_b1["delta_label"] = pd.Categorical(df_b1["delta_label"], categories=_b1_delta_order, ordered=True)

plot_b1 = (
    ggplot(df_b1, aes(x="t", y="p_identified", color="delta_label"))
    + geom_line(size=0.8)
    + scale_color_manual(values=_b1_colors)
    + labs(
        x="Sample size (t)",
        y="P(optimal policy identified by t)",
        color="Suboptimality gap",
        title="Demo 1b — Identification probability curve",
        subtitle=f"LIL CS,  M={_b1_M + 1} policies,  α={ALPHA}",
    )
    + theme_minimal()
    + theme(figure_size=(8, 4))
)
plot_b1.save("./figures/identification-probability.png", dpi=300, height=4, width=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2A: Single-run S_t heatmap
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  Demo 2A: Single-run S_t heatmap")
print("─" * 70)

gaps_2a = [0.05, 0.10, 0.15, 0.20]
_logging_gap = GAP * (1.0 - LOGGING_P)   # = 0.175, sits between Δ=0.15 and Δ=0.20
# Logging policy is always included. Inserted in true-value order so the heatmap
# displays policies from best (top) to worst (bottom) without resorting.
policy_probs_2a = (
    [1.0]
    + [(GAP - g) / GAP for g in gaps_2a[:3]]   # Δ = 0.05, 0.10, 0.15
    + [LOGGING_P]                               # Δ = 0.175 (logging)
    + [(GAP - g) / GAP for g in gaps_2a[3:]]   # Δ = 0.20
)
policy_names_2a = (
    ["Optimal"]
    + [f"Suboptimal {i+1} (Δ={g:.2f})" for i, g in enumerate(gaps_2a[:3])]
    + [f"Logging (Δ={_logging_gap:.3f})"]
    + [f"Suboptimal {len(gaps_2a[:3])+1+i} (Δ={g:.2f})" for i, g in enumerate(gaps_2a[3:])]
)
M_HEATMAP = len(policy_probs_2a)
alpha_2a = ALPHA / M_HEATMAP

rng_2a = np.random.default_rng(SEED + 1)
sim_2a = make_sim(rng_2a)
data_s = sim_2a.simulate(T_SINGLE)
pred_s = OLSRewardPredictor(actions=np.array([0, 1]))
pred_s.fit(data_s.contexts, data_s.actions, data_s.rewards)

# ── LIL heatmap ───────────────────────────────────────────────────────────────
all_L_s, all_U_s = [], []
for p in tqdm(policy_probs_2a, desc="Demo 2A (LIL): building CSs", unit="policy"):
    phi_l, phi_u = sim_2a.evaluate_policy(
        data_s, make_policy(p), kind="dr", reward_predictor=pred_s
    )
    L, U = LILConfidenceSequence(alpha=alpha_2a, k=K).bounds(phi_l, phi_u)
    all_L_s.append(L)
    all_U_s.append(U)

in_set_lil = PolicySelector.optimal_set(np.stack(all_L_s), np.stack(all_U_s))
tau_lil = PolicySelector.stopping_time(np.stack(all_L_s), np.stack(all_U_s))

t_idx = np.arange(0, T_SINGLE, T_STRIDE)

def _build_heat_df(in_set, policy_names):
    rows = []
    for i, name in enumerate(policy_names):
        for j in t_idx:
            rows.append({"t": j + 1, "policy": name,
                         "status": "Not eliminated" if in_set[i, j] else "Eliminated"})
    df = pd.DataFrame(rows)
    df["policy"] = pd.Categorical(df["policy"], categories=policy_names[::-1], ordered=True)
    return df

def _heat_plot(df):
    return (
        ggplot(df, aes(x="t", y="policy", fill="status"))
        + geom_tile(aes(width=T_STRIDE), height=0.9)
        + scale_fill_manual(values={"Not eliminated": "#d6604d", "Eliminated": "#aaaaaa"})
        + labs(
            x="Sample size (t)", y="Policies", fill="",
            # title="Progression of optimal policy set",
            # subtitle=(
            #     f"α={ALPHA}"
            # )
        )
        + theme_minimal()
        + theme(
            figure_size=(13, 4),
            legend_position="bottom",
            title=element_text(hjust=0.5)
        )
    )

df_heat_lil = _build_heat_df(in_set_lil, policy_names_2a)
plot_heat_lil = _heat_plot(df_heat_lil)

remaining_lil = [policy_names_2a[i] for i in range(M_HEATMAP) if in_set_lil[i, -1]]
print(f"  T={T_SINGLE}, M={M_HEATMAP}, α/m={alpha_2a:.4f}")
print(f"  LIL  — τ={tau_lil}  (T+1={T_SINGLE+1} = never stopped),  S_T: {remaining_lil}")

# ── Demo 2A: CS ribbon plot ───────────────────────────────────────────────────
rows_cs = []
for i, (name, p) in enumerate(zip(policy_names_2a, policy_probs_2a)):
    nu = true_value(p)
    for j in t_idx:
        rows_cs.append({
            "t": j + 1,
            "policy": name,
            "lower": all_L_s[i][j],
            "upper": all_U_s[i][j],
            "true_value": nu,
        })
df_cs_2a = pd.DataFrame(rows_cs)
df_cs_2a["policy"] = pd.Categorical(
    df_cs_2a["policy"], categories=policy_names_2a, ordered=True
)
df_nu_2a = pd.DataFrame({
    "policy": pd.Categorical(policy_names_2a, categories=policy_names_2a, ordered=True),
    "true_value": [true_value(p) for p in policy_probs_2a],
})

plot_cs_2a = (
    ggplot(df_cs_2a, aes(x="t"))
    + geom_ribbon(aes(ymin="lower", ymax="upper"), fill="#4393c3", alpha=0.4)
    + geom_line(aes(y="lower"), color="#2166ac", size=0.3)
    + geom_line(aes(y="upper"), color="#2166ac", size=0.3)
    + geom_hline(
        data=df_nu_2a,
        mapping=aes(yintercept="true_value"),
        linetype="dashed",
        color="#d6604d",
        size=0.8,
    )
    + facet_wrap("policy", ncol=3)
    + labs(
        x="Time step  t",
        y="Policy value  ν(π)",
        title="Demo 2A — Confidence sequences per policy (LIL CS, single run)",
        subtitle=(
            f"LIL CS,  α/m={alpha_2a:.4f},  T={T_SINGLE},  M={M_HEATMAP}"
            f"  |  dashed line = true ν(π)"
        ),
    )
    + theme_bw()
    + theme(figure_size=(12, 8))
)

# ── Betting heatmap ───────────────────────────────────────────────────────────
all_L_b, all_U_b = [], []
for p in tqdm(policy_probs_2a, desc="Demo 2A (Betting): building CSs", unit="policy"):
    phi_l, phi_u = sim_2a.evaluate_policy(
        data_s, make_policy(p), kind="dr", reward_predictor=pred_s
    )
    L, U = BettingConfidenceSequence(alpha=alpha_2a, k=K).bounds(phi_l, phi_u)
    all_L_b.append(L)
    all_U_b.append(U)

in_set_bet = PolicySelector.optimal_set(np.stack(all_L_b), np.stack(all_U_b))
tau_bet = PolicySelector.stopping_time(np.stack(all_L_b), np.stack(all_U_b))

df_heat_bet = _build_heat_df(in_set_bet, policy_names_2a)
plot_heat_bet = _heat_plot(df_heat_bet) + coord_cartesian(xlim=(0, 2500))
plot_heat_bet.save(
    "./figures/elimination-progression.png",
    dpi=500,
    height=3,
    width=6.5
)

remaining_bet = [policy_names_2a[i] for i in range(M_HEATMAP) if in_set_bet[i, -1]]
print(f"  Betting — τ={tau_bet}  (T+1={T_SINGLE+1} = never stopped),  S_T: {remaining_bet}")

# ── Demo 2A: Betting CS ribbon plot ──────────────────────────────────────────
rows_cs_b = []
for i, (name, p) in enumerate(zip(policy_names_2a, policy_probs_2a)):
    nu = true_value(p)
    for j in t_idx:
        rows_cs_b.append({
            "t": j + 1,
            "policy": name,
            "lower": all_L_b[i][j],
            "upper": all_U_b[i][j],
            "true_value": nu,
        })
df_cs_2a_b = pd.DataFrame(rows_cs_b)
df_cs_2a_b["policy"] = pd.Categorical(
    df_cs_2a_b["policy"], categories=policy_names_2a, ordered=True
)

plot_cs_2a_b = (
    ggplot(df_cs_2a_b, aes(x="t"))
    + geom_ribbon(aes(ymin="lower", ymax="upper"), fill="#d6604d", alpha=0.4)
    + geom_line(aes(y="lower"), color="#d6604d", size=0.3)
    + geom_line(aes(y="upper"), color="#d6604d", size=0.3)
    + geom_hline(
        data=df_nu_2a,
        mapping=aes(yintercept="true_value"),
        linetype="dashed",
        color="black",
        size=0.5,
    )
    + facet_wrap("policy", ncol=3)
    + labs(
        x="Sample size (t)",
        y="Policy value",
        # title="Progression of policy-value confidence sequences"
    )
    + coord_cartesian(xlim=(0, 2500), ylim=(0.4, 0.8))
    # + theme_bw()
    + theme_minimal()
    + theme(title=element_text(hjust=0.5))
)
plot_cs_2a_b.save(
    "./figures/cs-progression.png",
    dpi=300,
    height=3,
    width=8
)

# ── Demo 2A-E1: Confidence sequence width trajectory ─────────────────────────
# CS width W_t = U_t − L_t over time for each policy (LIL CS, single run).
# Shows convergence rate and how suboptimal policies are eliminated first.

_width_rows = []
for i, (name, p) in enumerate(zip(policy_names_2a, policy_probs_2a)):
    for j in t_idx:
        _width_rows.append({
            "t": j + 1,
            "policy": name,
            "width": float(all_U_s[i][j] - all_L_s[i][j]),
        })
df_width = pd.DataFrame(_width_rows)
df_width["policy"] = pd.Categorical(
    df_width["policy"], categories=policy_names_2a, ordered=True
)

plot_e1 = (
    ggplot(df_width, aes(x="t", y="width", color="policy"))
    + geom_line(size=0.6, alpha=0.9)
    + labs(
        x="Sample size (t)",
        y="CS width  [U_t − L_t]",
        color="Policy",
        title="Demo 2A — Confidence sequence width over time (LIL CS)",
        subtitle=f"Single trial,  α/m={alpha_2a:.4f},  T={T_SINGLE}",
    )
    + theme_minimal()
    + theme(figure_size=(10, 4))
)
plot_e1.save("./figures/cs-width.png", dpi=300, height=4, width=10)

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2B: Aggregate coverage P(π* ∈ S_t)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  Demo 2B: Aggregate coverage P(π* ∈ S_t)")
print("─" * 70)

gaps_cov = [0.10, 0.15, 0.20, 0.25]
# π* (index 0) + suboptimal policies + logging policy (always included).
probs_cov = [1.0] + [(GAP - g) / GAP for g in gaps_cov] + [LOGGING_P]
alpha_cov = ALPHA / len(probs_cov)

seeds_cov = [SEED + 100 + i for i in range(N_COV)]
rows_cov = []

# ── LIL coverage ──────────────────────────────────────────────────────────────
results_lil = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(_run_coverage_trial)(s, T_COV, probs_cov, alpha_cov, method="lil")
    for s in tqdm(seeds_cov, desc="Demo 2B: LIL coverage trials", unit="trial")
)
covered_lil = np.stack(results_lil)   # (N_COV, T_COV) bool
sim_cov_lil = covered_lil.all(axis=1).mean()
for t_i, cov in enumerate(covered_lil.mean(axis=0)):
    rows_cov.append({"t": t_i + 1, "coverage": cov, "method": "LIL"})
print(f"  LIL     — simultaneous coverage: {sim_cov_lil:.2%}  (target ≥ {1-ALPHA:.0%})")

# ── Betting coverage ───────────────────────────────────────────────────────────
results_bet = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(_run_coverage_trial)(s, T_COV, probs_cov, alpha_cov, method="betting")
    for s in tqdm(seeds_cov, desc="Demo 2B: Betting coverage trials", unit="trial")
)
covered_bet = np.stack(results_bet)   # (N_COV, T_COV) bool
sim_cov_bet = covered_bet.all(axis=1).mean()
for t_i, cov in enumerate(covered_bet.mean(axis=0)):
    rows_cov.append({"t": t_i + 1, "coverage": cov, "method": "Betting"})
print(f"  Betting — simultaneous coverage: {sim_cov_bet:.2%}  (target ≥ {1-ALPHA:.0%})")

# ── Hoeffding (PAC) coverage ──────────────────────────────────────────────────
# Hoeffding bounds are NOT anytime-valid: they are valid at a single fixed t,
# but when applied sequentially across all t they may exclude π* at early times.
results_hoe = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(_run_coverage_trial)(s, T_COV, probs_cov, alpha_cov, method="hoeffding")
    for s in tqdm(seeds_cov, desc="Demo 2B: Hoeffding coverage trials", unit="trial")
)
covered_hoe = np.stack(results_hoe)
sim_cov_hoe = covered_hoe.all(axis=1).mean()
for t_i, cov in enumerate(covered_hoe.mean(axis=0)):
    rows_cov.append({"t": t_i + 1, "coverage": cov, "method": "Hoeffding"})
print(f"  Hoeffding — simultaneous coverage: {sim_cov_hoe:.2%}  (target ≥ {1-ALPHA:.0%})")

print(f"\n  N={N_COV} trials,  T={T_COV},  M={len(probs_cov)},  α/m={alpha_cov:.4f}")

df_cov = pd.DataFrame(rows_cov)

plot_cov = (
    ggplot(df_cov, aes(x="t", y="coverage", color="method"))
    + geom_line(size=0.8)
    + geom_hline(yintercept=1.0 - ALPHA, linetype="dashed", color="black", size=0.9)
    + scale_color_manual(values={"LIL": "#4393c3", "Betting": "#d6604d", "Hoeffding": "#e6ab02"})
    + labs(
        x="Time step  t",
        y="P(π* ∈ S_t)",
        color="CS method",
        title="Demo 2B — Coverage of optimal policy in S_t",
        subtitle=(
            f"α={ALPHA},  α/m={alpha_cov:.4f},  {N_COV} trials  "
            f"|  LIL simultaneous coverage: {sim_cov_lil:.1%}"
            f",  Betting: {sim_cov_bet:.1%}"
            f",  Hoeffding: {sim_cov_hoe:.1%}"
        ),
    )
    + theme_bw()
    + theme(figure_size=(10, 5))
)
plot_cov.save("./figures/coverage.png", dpi=300, height=5, width=10)

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2C: Stopping time τ vs. partial-overlap failure in a subpopulation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  Demo 2C: Stopping time τ vs. logging-policy overlap")
print("─" * 70)

# Partial-overlap DGP: X ~ Uniform(0,1),  R = 0.5·X + 0.3·A + N(0, σ²).
# The problematic subpopulation (X < frac_bad) has degraded logging overlap
# controlled by p_bad; the rest use balanced logging (p=0.5, w≤2).
# Four severity levels × three bad-subpopulation fractions = 12 combinations.
#
# Gap for this DGP: GAP_OV = 0.70 - 0.40 = 0.30.
# p_sub_ov = 1 - DELTA_MIN_OV / GAP_OV.
_GAP_OV = 0.30
_ov_label_names = ["Mild", "Moderate", "High", "Severe"]
# Keys stored in the dataframe — plain severity names, no w_max annotation.
_ov_labels = {p_bad: name for name, p_bad in zip(_ov_label_names, P_BAD_GRID)}
_ov_order  = [_ov_labels[p] for p in P_BAD_GRID]
# Facet keys stored in the dataframe — simple "X%" strings.
# Display text is applied at plot-time via _frac_plot_labels so it can be
# changed without re-running the simulations.
_frac_keys = {f: f"{round(f * 100):.0f}%" for f in FRAC_BAD_GRID}

p_sub_ov    = 1.0 - DELTA_MIN_OV / _GAP_OV
policies_ov = [make_policy(1.0)] + [make_policy(p_sub_ov)] * (M_OV - 1)
alpha_ov    = ALPHA / M_OV

rows_ov = []
for frac_bad in FRAC_BAD_GRID:
    for p_bad in P_BAD_GRID:
        sev_label  = _ov_labels[p_bad]
        frac_label = _frac_keys[frac_bad]
        # Unique seed key per (p_bad, frac_bad) cell
        cell_key_ov = (int(round(p_bad * 10000)) * 1000
                       + int(round(frac_bad * 1000)) + 900000)
        seeds_ov = [cell_key_ov * 10000 + i for i in range(N_TRIALS_OV)]
        taus_ov = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_run_trial_overlap)(
                s, T_MAX_OV, policies_ov, alpha_ov, p_bad, frac_bad
            )
            for s in tqdm(
                seeds_ov,
                desc=f"Demo 2C: frac_bad={frac_bad:.2f}  p_bad={p_bad:.4f}",
                unit="trial",
            )
        )
        taus_arr = np.array(taus_ov)
        for tau in taus_ov:
            rows_ov.append({
                "overlap":   sev_label,
                "frac_bad":  frac_label,
                "tau":       tau,
                "p_bad":     p_bad,
            })
        print(
            f"  frac_bad={frac_bad:.2f}  p_bad={p_bad:.4f}  max_w={1/p_bad:.0f}"
            f"  mean τ={np.mean(taus_arr):.1f}"
            f"  P(censored)={np.mean(taus_arr == T_MAX_OV):.1%}"
        )

df_ov = pd.DataFrame(rows_ov)
df_ov["overlap"] = pd.Categorical(df_ov["overlap"], categories=_ov_order, ordered=True)
_frac_key_order = [_frac_keys[f] for f in FRAC_BAD_GRID]
df_ov["frac_bad"] = pd.Categorical(df_ov["frac_bad"], categories=_frac_key_order, ordered=True)

_ov_colors = {
    _ov_labels[P_BAD_GRID[0]]: "#1a9641",
    _ov_labels[P_BAD_GRID[1]]: "#4393c3",
    _ov_labels[P_BAD_GRID[2]]: "#f4a582",
    _ov_labels[P_BAD_GRID[3]]: "#d6604d",
}

# ── Facet display labels — edit here without re-running simulations ────────────
_frac_plot_labels = {
    "5%":  "5% of contexts in subgroup",
    "10%": "10% of contexts in subgroup",
    "20%": "20% of contexts in subgroup",
}

plot_ov = (
    ggplot(df_ov, aes(x="overlap", y="tau", fill="overlap"))
    + geom_violin(alpha=0.6, color="none")
    + geom_boxplot(width=0.12, fill="white", outlier_shape=None)
    + facet_wrap("frac_bad", nrow=3, scales="free_y",
                 labeller=labeller(frac_bad=lambda s: _frac_plot_labels.get(s, s)))
    + scale_fill_manual(values=_ov_colors)
    + labs(
        x="Overlap severity in high-risk subgroup",
        y="Stopping time (sample size)",
    )
    + theme_minimal()
    + theme(legend_position="none")
)
plot_ov.save(
    "./figures/weak-overlap.png",
    dpi=500,
    height=6,
    width=6,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 3: Stopping time τ vs. policy-class size M (B2 — Bonferroni scaling)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  Demo 3: Stopping time τ vs. policy-class size M (Bonferroni scaling)")
print("─" * 70)
print(f"  Fixed Δ_min={DELTA_B2:.2f},  LIL CS,  T_MAX={T_MAX}")

rows_b2 = []
for m_b2 in M_SWEEP:
    p_sub_b2 = (GAP - DELTA_B2) / GAP
    # Logging policy is always included, total policy count = m_b2 + 1.
    policies_b2 = (
        [make_policy(1.0)]
        + [make_policy(p_sub_b2)] * (m_b2 - 1)
        + [make_policy(LOGGING_P)]
    )
    alpha_b2 = ALPHA / (m_b2 + 1)
    cell_key_b2 = 700000 + m_b2
    seeds_b2 = [cell_key_b2 * 10000 + i for i in range(N_TRIALS_B2)]
    taus_b2 = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_run_trial_lil)(s, T_MAX, policies_b2, alpha_b2)
        for s in tqdm(seeds_b2, desc=f"Demo 3: M={m_b2}", unit="trial")
    )
    taus_arr_b2 = np.array(taus_b2)
    for tau in taus_b2:
        rows_b2.append({"M": m_b2, "tau": tau})
    print(
        f"  M={m_b2:3d}  α/m={alpha_b2:.5f}"
        f"  mean τ={np.mean(taus_arr_b2):7.1f}"
        f"  P(censored)={np.mean(taus_arr_b2 == T_MAX):.1%}"
    )

df_b2 = pd.DataFrame(rows_b2)

plot_b2 = (
    ggplot(df_b2, aes(x="M", y="tau"))
    + stat_summary(fun_y=np.mean, geom="line", color="#4393c3", size=1)
    + stat_summary(fun_y=np.mean, geom="point", color="#4393c3", size=2)
    + scale_x_log10()
    + labs(
        x="Policy-class size M  (log scale)",
        y="Mean stopping time (sample size)",
        title="Demo 3 — Stopping time vs. policy-class size (Bonferroni scaling)",
        subtitle=f"LIL CS,  Δ_min={DELTA_B2:.2f},  α={ALPHA}",
    )
    + theme_minimal()
    + theme(figure_size=(6, 4))
)
plot_b2.save("./figures/m-scaling.png", dpi=300, height=4, width=6)
print(f"  Saved ./figures/m-scaling.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 4: Nuisance model sensitivity — effect of reward predictor quality (C1)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 70)
print("  Demo 4: Nuisance model sensitivity — reward predictor quality")
print("─" * 70)
print(f"  Fixed Δ_min={DELTA_C1:.2f},  LIL CS,  T_MAX={T_MAX}")
print("  Predictors: Oracle (true μ)  vs.  OLS (linear)  vs.  Constant (mean)")

p_sub_c1 = (GAP - DELTA_C1) / GAP
policies_c1 = [make_policy(1.0)] + [make_policy(p_sub_c1)] + [make_policy(LOGGING_P)]
alpha_c1 = ALPHA / len(policies_c1)

rows_c1 = []
for ptype in ["oracle", "ols", "constant"]:
    # Deterministic per-type seed offset using hash for portability
    seed_offset_c1 = {"oracle": 800100, "ols": 800200, "constant": 800300}[ptype]
    seeds_c1 = [seed_offset_c1 + i for i in range(N_TRIALS_C1)]
    taus_c1 = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_run_trial_predictor)(s, T_MAX, policies_c1, alpha_c1, ptype)
        for s in tqdm(seeds_c1, desc=f"Demo 4: {ptype}", unit="trial")
    )
    taus_arr_c1 = np.array(taus_c1)
    for tau in taus_c1:
        rows_c1.append({"predictor": ptype, "tau": tau})
    print(
        f"  {ptype:<12}  mean τ={np.mean(taus_arr_c1):7.1f}"
        f"  P(censored)={np.mean(taus_arr_c1 == T_MAX):.1%}"
    )

df_c1 = pd.DataFrame(rows_c1)
_c1_label_map = {"oracle": "Oracle", "ols": "OLS", "constant": "Constant"}
_c1_order = ["Oracle", "OLS", "Constant"]
df_c1["predictor"] = pd.Categorical(
    df_c1["predictor"].map(_c1_label_map), categories=_c1_order, ordered=True
)
_c1_colors = {"Oracle": "#1a9641", "OLS": "#4393c3", "Constant": "#d6604d"}

plot_c1 = (
    ggplot(df_c1, aes(x="predictor", y="tau", fill="predictor"))
    + geom_violin(alpha=0.6, color="none")
    + geom_boxplot(width=0.12, fill="white", outlier_shape=None)
    + scale_fill_manual(values=_c1_colors)
    + labs(
        x="Reward predictor",
        y="Stopping time (sample size)",
        title="Demo 4 — Effect of reward predictor quality on stopping time",
        subtitle=(
            f"LIL CS,  Δ_min={DELTA_C1:.2f},  M={len(policies_c1)},  α={ALPHA}"
            "  |  validity holds for all predictor types (DR property)"
        ),
    )
    + theme_minimal()
    + theme(legend_position="none", figure_size=(6, 4))
)
plot_c1.save("./figures/nuisance-sensitivity.png", dpi=300, height=4, width=6)
print(f"  Saved ./figures/nuisance-sensitivity.png")

# ── Display plots (full mode only — show() blocks until window is closed) ──────
if FULL_RUN:
    print("\n" + "─" * 70)
    print("  Displaying plots (close each window to continue)…")
    print("─" * 70)

    print("\n  Demo 1: Stopping time grid…")
    plot_stop.show()

    print("\n  Demo 1b: Identification probability curve…")
    plot_b1.show()

    print("\n  Demo 2A: S_t heatmap (LIL)…")
    plot_heat_lil.show()

    print("\n  Demo 2A: S_t heatmap (Betting)…")
    plot_heat_bet.show()

    print("\n  Demo 2A: CS ribbon plot (LIL)…")
    plot_cs_2a.show()

    print("\n  Demo 2A: CS ribbon plot (Betting)…")
    plot_cs_2a_b.show()

    print("\n  Demo 2A-E1: CS width trajectory…")
    plot_e1.show()

    print("\n  Demo 2B: Coverage P(π* ∈ S_t)…")
    plot_cov.show()

    print("\n  Demo 2C: Stopping time vs. overlap…")
    plot_ov.show()

    print("\n  Demo 3: τ vs. M scaling…")
    plot_b2.show()

    print("\n  Demo 4: Nuisance sensitivity…")
    plot_c1.show()

print("\nDone.")
