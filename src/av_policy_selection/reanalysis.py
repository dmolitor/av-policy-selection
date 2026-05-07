"""
Anytime-valid optimal policy identification re-analysis of the infodemic study.

Re-analysis of: Offer-Westort et al., "Battling the
coronavirus 'infodemic' among social media users in Kenya and Nigeria,"
Nature Human Behaviour. https://doi.org/10.1038/s41562-023-01810-7

Data source
-----------
The file `cleaned-data_2023-03-28.csv` must be present in the data directory.
It is the cleaned replication dataset released by the original authors at:

    https://github.com/gsbDBI/infodemic-replication

Download the file from that repository (data/cleaned-data_2023-03-28.csv)
and place it in this project's data/ directory before running the analysis.

Policy class (8 constant respondent-level policies, all paired with H_control):
  π₀  Control              — arm 0  (no treatment)
  π₁  Accuracy nudge       — arm 5
  π₂  Deliberation nudge   — arm 6
  π₃  Emotion suppression  — arm 7
  π₄  Pledge               — arm 8
  π₅  AfricaCheck tips     — arm 9
  π₆  Facebook tips        — arm 10
  π₇  Video training       — arm 11

Arms 7, 9, 11 have zero or near-zero propensity in batch 5 (the evaluation phase),
but are all eliminated before batch 5 begins (t < 4762), so their elimination
decisions are based entirely on valid anytime-valid inference from batches 1–4.
Arms 6 and 8 have zero batch-5 propensity but are not eliminated before batch 5;
pseudo-outcomes for those observations are model-based (OLS extrapolation).

Note: the paper's targeted policies (LTP, RTP) lie out of scope for this analysis.
Their advantage over constant accuracy nudge (+0.005 on [0,1] scale) is too small
to detect under sequential inference with N=15,292; fixed-sample inference in the
original paper required p-values near 0.004 to claim significance, and sequential
guarantees require substantially larger samples.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from av_policy_selection import BettingConfidenceSequence, OLSRewardPredictor
from av_policy_selection.load_data import (
    POLICY_ARMS,
    POLICY_NAMES,
    load_infodemic,
)

import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from utils import build_heat_df, heat_plot

# ── Parameters ────────────────────────────────────────────────────────────────
_ALPHA = 0.05
_K     = 0.0


def run_reanalysis(data_dir: Path | str, fig_dir: Path | str) -> None:
    """Run the full infodemic re-analysis and save figures.

    Parameters
    ----------
    data_dir : path to directory containing cleaned-data_2023-03-28.csv
    fig_dir  : directory where figures are saved (created if missing)
    """
    data_dir = Path(data_dir)
    fig_dir  = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)

    ALPHA    = _ALPHA
    K        = _K
    M        = len(POLICY_NAMES)
    alpha_policy = ALPHA / M

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    contexts, actions, rewards, propensities = load_infodemic(data_dir)
    N = len(rewards)
    print(f"  N={N}, M={M}, α_policy={alpha_policy:.4f}")

    # ── Fit reward predictor ──────────────────────────────────────────────────
    print("Fitting reward predictor...")
    all_arms = np.arange(40)
    aipw_pred = OLSRewardPredictor(actions=all_arms)
    aipw_pred.fit(contexts, actions, rewards)
    r_hat_obs = aipw_pred.predict(contexts, actions)

    # ── Compute AIPW pseudo-outcomes ──────────────────────────────────────────
    print("Computing AIPW pseudo-outcomes...")
    phi_drl_all = np.zeros((N, M))
    phi_dru_all = np.zeros((N, M))

    for j, arm in enumerate(POLICY_ARMS):
        prescribed = np.full(N, arm)
        indicator  = (actions == prescribed).astype(float)
        prop       = np.where(propensities > 0, propensities, 1.0)
        weights    = indicator / prop
        r_hat_pi   = aipw_pred.predict(contexts, prescribed)
        residual   = rewards - r_hat_obs
        phi_drl_all[:, j] = np.clip(weights * residual + r_hat_pi, 0.0, 1.0)
        phi_dru_all[:, j] = np.clip(-weights * residual + (1.0 - r_hat_pi), 0.0, 1.0)

    # ── Run confidence sequences ─────────────────────────────────────
    print("Running confidence sequences...")
    cs = BettingConfidenceSequence(alpha=alpha_policy, k=K)

    lower_all = np.zeros((N, M))
    upper_all = np.zeros((N, M))
    for j in range(M):
        lower_all[:, j] = cs.lower(phi_drl_all[:, j])
        upper_all[:, j] = cs.upper(phi_dru_all[:, j])

    # ── Elimination set S_t ───────────────────────────────────────────────────
    in_set = upper_all >= lower_all.max(axis=1, keepdims=True)

    print(f"  Final S_t: {[POLICY_NAMES[j] for j in range(M) if in_set[-1, j]]}")
    for j, name in enumerate(POLICY_NAMES):
        elim = np.where(~in_set[:, j])[0]
        t_elim = int(elim[0]) + 1 if len(elim) else None
        print(f"  {name:25s}: eliminated at t={t_elim}")

    # ── Elimination heatmap ───────────────────────────────────────────────────
    print("Building figures...")
    t_vals = np.arange(1, N + 1)

    # Sort policies: never-eliminated first, then by elimination time descending
    _never_elim, _eliminated = [], []
    for j, name in enumerate(POLICY_NAMES):
        elim_idx = np.where(~in_set[:, j])[0]
        if len(elim_idx) == 0:
            _never_elim.append((j, name))
        else:
            _eliminated.append((j, name, int(elim_idx[0])))
    _eliminated_sorted = sorted(_eliminated, key=lambda x: x[2], reverse=True)
    _ordered_idx   = [x[0] for x in _never_elim] + [x[0] for x in _eliminated_sorted]
    _ordered_names = [x[1] for x in _never_elim] + [x[1] for x in _eliminated_sorted]
    in_set_sorted  = in_set[:, _ordered_idx]

    T_STRIDE    = max(1, N // 100)
    col_indices = list(range(0, N, T_STRIDE))
    t_values    = [t_vals[i] for i in col_indices]
    df_heat     = build_heat_df(in_set_sorted.T, _ordered_names, col_indices, t_values)
    fig_heat    = heat_plot(df_heat, T_STRIDE, xlim=(0, N))
    fig_heat.save(str(fig_dir / "figure3.png"), dpi=300, height=4, width=7)

    print("Done. Figures saved to", fig_dir)


if __name__ == "__main__":
    _data_dir = Path(__file__).parents[2] / "data"
    _fig_dir  = Path(__file__).parents[2] / "figures"
    run_reanalysis(_data_dir, _fig_dir)
