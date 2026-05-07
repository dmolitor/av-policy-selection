"""
Replication script for the empirical results in
"Anytime-valid Optimal Policy Identification" (Molitor, NeurIPS 2026).

Running this file produces the four figures in Section 4 of the paper:

  figures/figure1a.png  — single-run S_t heatmap     (illustrative example)
  figures/figure1b.png  — single-run CS ribbons      (illustrative example)
  figures/figure2.png   — sample-savings curve       (power-analysis overshoot)
  figures/figure3.png   — infodemic S_t heatmap      (Offer-Westort et al. 2024)

Synthetic DGP for Figures 1 and 2:
  Contexts X ~ Uniform[0, 1]^3.
  Logging policy h(A=1 | X) = clip(sigma(w^T X), 0.10, 0.90); fixed w drawn at seed 1.
  Conditional mean mu(X, A) = BETA[A] . [1, X], with marginal means EV[0]=0.40
  and EV[1]=0.70 (gap GAP_DGP=0.30). Rewards R ~ Beta(mu * KAPPA, (1-mu) * KAPPA).
  Candidate policy pi_j is the action mixture P(A=1 | pi_j) = p_j; the optimal
  candidate sets p_0 = 1 (always treat), and suboptimal candidates use p_j < 1
  to induce a chosen suboptimality gap.

Figure 2 (sample savings under power-analysis overshoot):
  For each powered-for gap Delta_powered, we find the oracle fixed sample size
  N_90 — the smallest N at which the PrPL fixed-sample CI of Luedtke & Soni
  (2024, Corollary 1) identifies pi^* with probability >= 0.90 — by binary
  search. We then simulate the anytime-valid analogue (PrPL CS) at true gaps
  Delta_true = c * Delta_powered for c in C_GRID, with a budget cap of 5*N_90,
  and report 1 - E[tau] / N_90 averaged across powered-for gaps.

Modes:
  uv run python simulations.py             # quick test mode
  FULL_RUN=1 uv run python simulations.py  # full (paper) settings
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
import mizani
from joblib import Parallel, delayed
from scipy.special import expit as _sigmoid
from tqdm import tqdm

from av_policy_selection import (
    LILConfidenceSequence,
    OLSRewardPredictor,
    PolicySelector,
    PrPLConfidenceInterval,
    PrPLConfidenceSequence,
)
from av_policy_selection.reanalysis import run_reanalysis
from utils import build_heat_df, heat_plot, make_cs_ribbon_plot


# ── Run mode ─────────────────────────────────────────────────────────────────
# FULL_RUN=1 reproduces the paper's settings; default is a much smaller test run.
FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
FIG_DIR  = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Shared parameters ────────────────────────────────────────────────────────
ALPHA     = 0.10              # global error level
K_TRUNC   = 1.0               # CS truncation parameter k (see paper, Sec. 2)
D_CONTEXT = 3                 # context dimension
K_ARMS    = 2                 # number of actions
KAPPA     = 1.0               # Beta concentration (variance = mu(1-mu)/(KAPPA+1))

# Conditional-mean coefficients: mu(X, A) = BETA[A] . [1, X1, X2, X3].
# Row 0 is the control arm (A=0), row 1 the treatment arm (A=1).
BETA      = np.array([
    [0.25, 0.10, 0.10, 0.10],   # mean = 0.40
    [0.55, 0.10, 0.10, 0.10],   # mean = 0.70
])
EV        = BETA[:, 0] + 0.5 * BETA[:, 1:].sum(axis=1)   # marginal arm means
GAP_DGP   = float(EV[1] - EV[0])                         # = 0.30

# Logistic logging policy: P(A=1|X) = clip(sigma(_LOG_W . X), LOG_CLIP, 1-LOG_CLIP).
# Clipping caps the maximum IPS weight at 1/LOG_CLIP = 10.
_LOG_W    = np.random.default_rng(1).standard_normal(D_CONTEXT)
LOG_CLIP  = 0.10

# ── Figure 1 (illustrative example) ──────────────────────────────────────────
SEED_DEMO       = 42            # fixed seed for reproducibility
M_DEMO          = 5             # candidate policies (1 optimal + 4 suboptimal)
DELTA_MIN_DEMO  = 0.05          # gap of the closest suboptimal policy
DELTA_STEP_DEMO = 0.01          # gap increment between successive suboptimals

# ── Figure 2 (sample-savings) ────────────────────────────────────────────────
# c = ratio (true gap) / (powered-for gap)
C_GRID = [1.0, 1.25, 1.50, 1.75, 2.00, 2.5, 3]

if FULL_RUN:
    N_TRIALS_FIXED = 100         # trials per binary-search step for N_90
    N_TRIALS_F2    = 500         # PrPL CS trials per (Delta_powered, c) cell
    T_SINGLE       = 5_000       # Figure 1 horizon
    T_STRIDE_DEMO  = 100         # subsampling stride for Figure 1 ribbons/heatmap
    M_F2           = 10          # policy-class size used in Figure 2
    GAP_GRID_F2    = [0.02, 0.05, 0.10, 0.15, 0.20]
    T_MAX          = 100_000     # budget cap for binary search of N_90
    N_JOBS         = -1
else:
    N_TRIALS_FIXED = 10
    N_TRIALS_F2    = 15
    T_SINGLE       = 300
    T_STRIDE_DEMO  = 30
    M_F2           = 5
    GAP_GRID_F2    = [0.05, 0.10, 0.20]
    T_MAX          = 50_000
    N_JOBS         = 1


# ════════════════════════════════════════════════════════════════════════════
# DGP helpers
# ════════════════════════════════════════════════════════════════════════════

def _mu_fn(X, A):
    """Conditional reward mean E[R | X, A] = BETA[A] . [1, X]."""
    Xaug = np.c_[np.ones(len(X)), X]
    return (BETA[A.astype(int)] * Xaug).sum(axis=1).clip(0.0, 1.0)


def _log_prop(X):
    """Logistic logging-policy treatment propensity P(A=1 | X)."""
    return np.clip(_sigmoid(X @ _LOG_W), LOG_CLIP, 1.0 - LOG_CLIP)


def _policy_probs(M, gap):
    """Mixing probabilities p_j for M candidate policies under a fixed gap.

    Policy 0 is optimal (always plays A=1). The remaining M-1 policies share
    the same suboptimality gap = `gap` from the optimum, achieved by mixing
    arms with probability p_sub of treating.
    """
    p_sub = (GAP_DGP - gap) / GAP_DGP
    return np.array([1.0] + [p_sub] * (M - 1))


# ════════════════════════════════════════════════════════════════════════════
# Pseudo-outcome and CS workers
# ════════════════════════════════════════════════════════════════════════════

def _aipw_outcomes(contexts, actions, rewards, prop_matrix, num_arms):
    """AIPW (doubly-robust) pseudo-outcomes for every constant arm-policy.

    Returns
    -------
    phi_drl : ndarray, shape (T, num_arms)
        Lower DR pseudo-outcome (E[. | H_{t-1}] = nu(pi_a)).
    phi_dru : ndarray, shape (T, num_arms)
        Upper (mirrored) DR pseudo-outcome.
    """
    predictor = OLSRewardPredictor(actions=np.arange(num_arms))
    predictor.fit(contexts, actions, rewards)
    r_hat_obs = predictor.predict(contexts, actions)

    T = len(rewards)
    phi_drl = np.zeros((T, num_arms))
    phi_dru = np.zeros((T, num_arms))
    for a in range(num_arms):
        h_a   = np.where(prop_matrix[:, a] > 0, prop_matrix[:, a], 1.0)
        w_a   = (actions == a).astype(float) / h_a
        r_a   = predictor.predict(contexts, np.full(T, a))
        resid = rewards - r_hat_obs
        phi_drl[:, a] = np.clip(w_a * resid + r_a, 0.0, 1.0)
        phi_dru[:, a] = np.clip(-w_a * resid + (1.0 - r_a), 0.0, 1.0)
    return phi_drl, phi_dru


def _synthetic_aipw(rng, M, gap, T):
    """Generate T synthetic observations and return AIPW pseudo-outcomes for M policies.

    The AIPW pseudo-outcome for the mixing policy pi_j is the convex combination
    p_j * phi(arm 1) + (1 - p_j) * phi(arm 0) of the per-arm pseudo-outcomes.
    """
    X       = rng.uniform(0, 1, size=(T, D_CONTEXT))
    p1      = _log_prop(X)
    prop    = np.c_[1.0 - p1, p1]
    actions = rng.binomial(1, p1).astype(int)
    mu_obs  = _mu_fn(X, actions)
    rewards = rng.beta(mu_obs * KAPPA, (1.0 - mu_obs) * KAPPA)

    phi_drl_arms, phi_dru_arms = _aipw_outcomes(X, actions, rewards, prop, K_ARMS)

    probs   = _policy_probs(M, gap)
    phi_drl = np.stack(
        [probs[j] * phi_drl_arms[:, 1] + (1 - probs[j]) * phi_drl_arms[:, 0]
         for j in range(M)],
        axis=1,
    )
    phi_dru = np.stack(
        [probs[j] * phi_dru_arms[:, 1] + (1 - probs[j]) * phi_dru_arms[:, 0]
         for j in range(M)],
        axis=1,
    )
    return phi_drl, phi_dru


def _grid_prpl_trial(seed, M, gap, N, alpha_policy):
    """One fixed-N PrPL CI trial; True iff S_N = {pi^*}."""
    rng              = np.random.default_rng(seed)
    phi_drl, phi_dru = _synthetic_aipw(rng, M, gap, N)
    ci               = PrPLConfidenceInterval(alpha=alpha_policy)
    lower            = np.array([ci.lower(phi_drl[:, a]) for a in range(M)])
    upper            = np.array([ci.upper(phi_dru[:, a]) for a in range(M)])
    in_set           = upper >= lower.max()
    return bool(in_set[0] and in_set.sum() == 1)


def _prpl_cs_trial_f2(seed, M, gap, T, alpha_policy):
    """One PrPL CS trial; returns the stopping time tau (capped at T)."""
    rng              = np.random.default_rng(seed)
    phi_drl, phi_dru = _synthetic_aipw(rng, M, gap, T)
    cs               = PrPLConfidenceSequence(alpha=alpha_policy)
    lower = np.stack([cs.lower(phi_drl[:, a]) for a in range(M)])
    upper = np.stack([cs.upper(phi_dru[:, a]) for a in range(M)])
    return min(PolicySelector.stopping_time(lower, upper), T)


def _find_n90(trial_fn, trial_kwargs, seed_base=0):
    """Binary search for the smallest N s.t. P(S_N = {pi^*}) >= 0.90."""
    lo, hi = 10, T_MAX
    while lo < hi:
        mid   = (lo + hi) // 2
        seeds = [seed_base * 10 + i for i in range(N_TRIALS_FIXED)]
        hits  = Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(trial_fn)(s, **{**trial_kwargs, "N": mid}) for s in seeds
        )
        if np.mean(hits) >= 0.90:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — single-run illustrative example
# ════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  Figure 1: Single-run S_t heatmap + CS ribbons (LIL CS)")
print("─" * 70)

# Five candidate policies + the logging policy. Suboptimal gaps step by 0.01.
_demo_gaps  = [0.0] + [DELTA_MIN_DEMO + k * DELTA_STEP_DEMO for k in range(M_DEMO - 1)]
_demo_probs = [(GAP_DGP - g) / GAP_DGP for g in _demo_gaps]
print(f"  M={M_DEMO} candidates (Δ_min={DELTA_MIN_DEMO}) + logging policy, "
      f"T={T_SINGLE}, seed={SEED_DEMO}")

# Generate the bandit data once and compute per-arm AIPW pseudo-outcomes.
_rng_demo  = np.random.default_rng(87934)
_X_demo    = _rng_demo.uniform(0, 1, size=(T_SINGLE, D_CONTEXT))
_p1_demo   = _log_prop(_X_demo)
_prop_demo = np.c_[1.0 - _p1_demo, _p1_demo]
_act_demo  = _rng_demo.binomial(1, _p1_demo).astype(int)
_mu_demo   = _mu_fn(_X_demo, _act_demo)
_rew_demo  = _rng_demo.beta(_mu_demo * KAPPA, (1.0 - _mu_demo) * KAPPA)
phi_drl_arms_demo, phi_dru_arms_demo = _aipw_outcomes(
    _X_demo, _act_demo, _rew_demo, _prop_demo, K_ARMS
)

# Candidate policy pseudo-outcomes are convex combinations of arm-level ones.
phi_drl_cand = np.stack(
    [_demo_probs[j] * phi_drl_arms_demo[:, 1] + (1 - _demo_probs[j]) * phi_drl_arms_demo[:, 0]
     for j in range(M_DEMO)], axis=1,
)
phi_dru_cand = np.stack(
    [_demo_probs[j] * phi_dru_arms_demo[:, 1] + (1 - _demo_probs[j]) * phi_dru_arms_demo[:, 0]
     for j in range(M_DEMO)], axis=1,
)

# Logging policy pseudo-outcomes use the realised treatment propensity p1(X_t).
phi_drl_log_vec = _p1_demo * phi_drl_arms_demo[:, 1] + (1 - _p1_demo) * phi_drl_arms_demo[:, 0]
phi_dru_log_vec = _p1_demo * phi_dru_arms_demo[:, 1] + (1 - _p1_demo) * phi_dru_arms_demo[:, 0]
# True logging-policy value: v(pi_log) = EV[0] + GAP_DGP * E_X[p1(X)].
_v_log = float(EV[0] + GAP_DGP * np.mean(_p1_demo))

# LIL CSs over all M_DEMO+1 policies with Bonferroni correction.
_alpha_demo = ALPHA / (M_DEMO + 1)
_cs_demo    = LILConfidenceSequence(alpha=_alpha_demo, k=K_TRUNC)

phi_drl_all = np.hstack([phi_drl_cand, phi_drl_log_vec[:, None]])    # (T, M+1)
phi_dru_all = np.hstack([phi_dru_cand, phi_dru_log_vec[:, None]])
lower_demo  = np.stack([_cs_demo.lower(phi_drl_all[:, a]) for a in range(M_DEMO + 1)])
upper_demo  = np.stack([_cs_demo.upper(phi_dru_all[:, a]) for a in range(M_DEMO + 1)])

_policy_names_demo = (
    ["Optimal"]
    + [f"Suboptimal {j} (Δ={_demo_gaps[j]:.2f})" for j in range(1, M_DEMO)]
    + ["Logging"]
)
_true_vals_demo = (
    [float(EV[1])]
    + [float(EV[1] - g) for g in _demo_gaps[1:]]
    + [_v_log]
)

_tau_demo       = PolicySelector.stopping_time(lower_demo, upper_demo)
_in_set_all     = PolicySelector.optimal_set(lower_demo, upper_demo)
_remaining_demo = [_policy_names_demo[i] for i in range(M_DEMO + 1) if _in_set_all[i, -1]]
print(f"  τ={_tau_demo}  (T+1={T_SINGLE+1} = never stopped)")
print(f"  S_T: {_remaining_demo}")
print(f"  True v(π_log) ≈ {_v_log:.3f}")

# Heatmap (Figure 1a) — subsample at stride points to keep output compact.
_t_idx_demo  = np.arange(0, T_SINGLE, T_STRIDE_DEMO)
df_heat_demo = build_heat_df(
    _in_set_all, _policy_names_demo,
    col_indices=_t_idx_demo, t_values=_t_idx_demo + 1,
)
plot_heat_demo = heat_plot(df_heat_demo, T_STRIDE_DEMO, xlim=(0, T_SINGLE))
plot_heat_demo.save(str(FIG_DIR / "figure1a.png"), dpi=300, height=3, width=7)
print(f"  Saved: {FIG_DIR / 'figure1a.png'}")

# CS ribbons (Figure 1b) — one panel per policy.
_cs_rows_demo = [
    {"t": int(j + 1), "policy": name,
     "lower": float(lower_demo[i, j]), "upper": float(upper_demo[i, j])}
    for i, name in enumerate(_policy_names_demo) for j in _t_idx_demo
]
df_cs_demo = pd.DataFrame(_cs_rows_demo)
df_cs_demo["policy"] = pd.Categorical(
    df_cs_demo["policy"], categories=_policy_names_demo, ordered=True
)
df_nu_demo = pd.DataFrame({
    "policy": pd.Categorical(_policy_names_demo, categories=_policy_names_demo, ordered=True),
    "true_value": _true_vals_demo,
})
_ylim_demo = (
    max(0.0, min(_true_vals_demo) - 0.15),
    min(1.0, float(EV[1]) + 0.15),
)
plot_cs_demo = make_cs_ribbon_plot(
    df_cs_demo, df_nu_demo, xlim=T_SINGLE, ylim=_ylim_demo, hline_size=0.6
)
plot_cs_demo.save(str(FIG_DIR / "figure1b.png"), dpi=300, height=3, width=6)
print(f"  Saved: {FIG_DIR / 'figure1b.png'}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — sample savings under power-analysis overshoot
# ════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  Figure 2: Mean sample savings 1 - E[τ]/N_90 vs c (gap multiplier)")
print("─" * 70)
print(f"  M={M_F2}  GAP_GRID={GAP_GRID_F2}  C_GRID={C_GRID}")
print(f"  N_TRIALS_F2={N_TRIALS_F2}  N_TRIALS_FIXED={N_TRIALS_FIXED}  T_MAX={T_MAX:,}")
print(f"  Cells with Δ_true > GAP_DGP={GAP_DGP:.2f} are skipped (DGP infeasible).")

_alpha_policy_f2 = ALPHA / M_F2
_rows_f2         = []

for _gap_pow in tqdm(GAP_GRID_F2, desc="Figure 2 grid"):
    _N_90 = _find_n90(
        trial_fn=_grid_prpl_trial,
        trial_kwargs={"M": M_F2, "gap": _gap_pow, "alpha_policy": _alpha_policy_f2},
        seed_base=M_F2 * 100 + int(_gap_pow * 10000),
    )
    print(f"  Δ_pow={_gap_pow:.2f}  N_90={_N_90:,}")

    for _c in C_GRID:
        _gap_true = _c * _gap_pow
        if _gap_true > GAP_DGP:
            print(f"    c={_c}  Δ_true={_gap_true:.3f} > GAP_DGP={GAP_DGP:.2f} — skipped")
            continue

        # Cap each trial at 5*N_90 so τ > N_90 is recorded as negative savings.
        _budget = 5 * _N_90
        _taus   = np.array(Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(_prpl_cs_trial_f2)(s, M_F2, _gap_true, _budget, _alpha_policy_f2)
            for s in range(N_TRIALS_F2)
        ))

        _savings  = 1.0 - _taus / _N_90
        _mean_sav = float(np.mean(_savings))
        _se_sav   = float(np.std(_savings, ddof=1) / np.sqrt(N_TRIALS_F2))
        _rows_f2.append({
            "delta_powered": _gap_pow,
            "delta_true":    float(_gap_true),
            "c":             _c,
            "N_90":          _N_90,
            "mean_tau":      float(np.mean(_taus)),
            "mean_savings":  _mean_sav,
            "se_savings":    _se_sav,
        })

df_f2 = pd.DataFrame(_rows_f2)
DATA_DIR.mkdir(exist_ok=True)
df_f2.to_csv(DATA_DIR / "fig2_savings.csv", index=False)

# Aggregate across powered-for gaps: average savings at each c, ±1 SE band.
_df_f2_agg = (
    df_f2.groupby("c")["mean_savings"]
         .agg(mean_savings="mean", se_savings=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
         .reset_index()
)
_df_f2_agg["savings_lo"] = _df_f2_agg["mean_savings"] - _df_f2_agg["se_savings"]
_df_f2_agg["savings_hi"] = _df_f2_agg["mean_savings"] + _df_f2_agg["se_savings"]

# Plot only c > 1 (savings vs overshoot).
_plot_f2 = (
    pn.ggplot(_df_f2_agg[_df_f2_agg["c"] > 1], pn.aes(x="c", y="mean_savings"))
    + pn.geom_line()
    + pn.geom_point()
    + pn.geom_ribbon(pn.aes(ymin="savings_lo", ymax="savings_hi"), alpha=0.3)
    + pn.scale_x_continuous(breaks=[1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])
    + pn.scale_y_continuous(labels=mizani.labels.percent)
    + pn.labs(x="True Δ ÷ Powered-for Δ", y="Mean sample savings")
    + pn.theme_minimal()
    + pn.theme(panel_grid_minor=pn.element_blank())
)
_plot_f2.save(str(FIG_DIR / "figure2.png"), dpi=300, width=5, height=3)
print(f"  Saved: {FIG_DIR / 'figure2.png'}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Offer-Westort et al. (2024) infodemic re-analysis
# ════════════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  Figure 3: Infodemic re-analysis (saves to figures/figure3.png)")
print("─" * 70)

if (DATA_DIR / "cleaned-data_2023-03-28.csv").exists():
    run_reanalysis(DATA_DIR, FIG_DIR)
else:
    print(f"  Data not found at {DATA_DIR}/cleaned-data_2023-03-28.csv — skipping.")
    print("  Download from: https://github.com/gsbDBI/infodemic-replication")
    print("  and place cleaned-data_2023-03-28.csv in the data/ directory.")

print("=" * 70)
print(f"  Done. Figures saved to {FIG_DIR}")
print("=" * 70)
