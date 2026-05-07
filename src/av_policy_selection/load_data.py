"""
Data loading and preprocessing for the infodemic re-analysis.

Source: Offer-Westort, et al., "Battling the coronavirus
'infodemic' among social media users in Kenya and Nigeria," Nature Human Behaviour.
Data repository: https://github.com/gsbDBI/infodemic-replication
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Arm encoding from the original experiment (40 arms: 5 headline × 8 respondent)
ARM_ENCODING = {
    "H_control_R_control": 0,
    "H_factcheck_R_control": 1,
    "H_more_info_R_control": 2,
    "H_real_info_R_control": 3,
    "H_related_R_control": 4,
    "H_control_R_accuracy": 5,
    "H_control_R_deliberation": 6,
    "H_control_R_emotion": 7,
    "H_control_R_pledge": 8,
    "H_control_R_tips_africacheck": 9,
    "H_control_R_tips_facebook": 10,
    "H_control_R_video": 11,
    "H_factcheck_R_accuracy": 12,
    "H_factcheck_R_deliberation": 13,
    "H_factcheck_R_emotion": 14,
    "H_factcheck_R_pledge": 15,
    "H_factcheck_R_tips_africacheck": 16,
    "H_factcheck_R_tips_facebook": 17,
    "H_factcheck_R_video": 18,
    "H_more_info_R_accuracy": 19,
    "H_more_info_R_deliberation": 20,
    "H_more_info_R_emotion": 21,
    "H_more_info_R_pledge": 22,
    "H_more_info_R_tips_africacheck": 23,
    "H_more_info_R_tips_facebook": 24,
    "H_more_info_R_video": 25,
    "H_real_info_R_accuracy": 26,
    "H_real_info_R_deliberation": 27,
    "H_real_info_R_emotion": 28,
    "H_real_info_R_pledge": 29,
    "H_real_info_R_tips_africacheck": 30,
    "H_real_info_R_tips_facebook": 31,
    "H_real_info_R_video": 32,
    "H_related_R_accuracy": 33,
    "H_related_R_deliberation": 34,
    "H_related_R_emotion": 35,
    "H_related_R_pledge": 36,
    "H_related_R_tips_africacheck": 37,
    "H_related_R_tips_facebook": 38,
    "H_related_R_video": 39,
}

# All 8 respondent-level constant policies (paired with headline control H_control).
# Arms 0, 5–11 correspond to H_control paired with each respondent-level intervention.
POLICY_ARMS = [0, 5, 6, 7, 8, 9, 10, 11]

# Full policy names in display order (matching POLICY_ARMS index-for-index)
POLICY_NAMES = [
    "Control",
    "Accuracy nudge",
    "Deliberation nudge",
    "Emotion suppression",
    "Pledge",
    "AfricaCheck tips",
    "Facebook tips",
    "Video training",
]

# Covariates used for reward prediction (from gen_probabilities.py)
XVARS = [
    "male", "age", "age_flag", "age_check_flag", "ed", "ed_flag", "urban",
    "rel_christian", "rel_muslim", "denom_pentecostal", "religiosity",
    "religiosity_flag", "locus", "locus_flag", "science", "science_flag",
    "dli", "fb_post", "fb_post_flag", "fb_msg", "fb_msg_flag", "crt",
    "hhi", "hhi_flag", "cash", "hh", "hh_flag", "pol", "cov_concern",
    "cov_concern_flag", "cov_efficacy", "cov_efficacy_flag", "nigeria",
    "strat_send_false0", "strat_send_false1", "strat_send_false2",
    "strat_send_true0", "strat_send_true1", "strat_send_true2",
    "strat_timeline_false0", "strat_timeline_false1", "strat_timeline_false2",
    "strat_timeline_true0", "strat_timeline_true1", "strat_timeline_true2",
]

# Y range from paper: Y = -M_post + 0.5*T_post, M ∈ {0..4}, T ∈ {0..4}
# Actual observed range: [-4, 2]; scale to [0, 1]
Y_MIN, Y_MAX = -4.0, 2.0


def load_infodemic(
    data_dir: str | Path,
    batches: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the infodemic dataset.

    Returns observations sorted by batch order (batches 1–5, then row order
    within batch), excluding attrited participants.

    Parameters
    ----------
    data_dir : path to directory containing cleaned-data_2023-03-28.csv

    Returns
    -------
    contexts : ndarray, shape (N, P)
        Covariate matrix with NaN-imputed columns (mean imputation).
    actions : ndarray, shape (N,), dtype int
        Arm index 0–39 for each observation.
    rewards : ndarray, shape (N,)
        Composite discernment outcome Y scaled to [0, 1].
    propensities : ndarray, shape (N,)
        Logged propensity P(A_i | X_i) from the adaptive experiment.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "cleaned-data_2023-03-28.csv")

    # Keep non-attrited only; optionally filter to specific batches
    df = df[df["attrited"] == 0].copy()
    if batches is not None:
        df = df[df["batch"].isin(batches)].copy()
    df = df.sort_values("batch", kind="stable").reset_index(drop=True)

    # Arm index
    df["arm_idx"] = df["W"].map(ARM_ENCODING)

    # Scaled reward
    df["reward"] = (df["Y"] - Y_MIN) / (Y_MAX - Y_MIN)

    # Propensity for the assigned arm
    prob_cols = [f"probs_{i}" for i in range(40)]
    probs_matrix = df[prob_cols].to_numpy()
    arm_indices = df["arm_idx"].to_numpy(dtype=int)
    propensities = probs_matrix[np.arange(len(df)), arm_indices]

    # Contexts: mean-impute missing values
    ctx = df[XVARS].to_numpy(dtype=float)
    col_means = np.nanmean(ctx, axis=0)
    nan_mask = np.isnan(ctx)
    ctx[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return ctx, arm_indices, df["reward"].to_numpy(), propensities


if __name__ == "__main__":
    data_dir = Path(__file__).parents[2] / "data"
    contexts, actions, rewards, propensities = load_infodemic(data_dir)
    N, P = contexts.shape
    print(f"N={N}, P={P}")
    print(f"Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"Propensity range: [{propensities.min():.4f}, {propensities.max():.4f}]")
    print(f"Propensity mean: {propensities.mean():.4f}")
    print(f"Actions unique: {sorted(set(actions.tolist()))}")
    print(f"NaN in contexts: {np.isnan(contexts).any()}")
