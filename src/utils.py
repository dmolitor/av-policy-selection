"""
Plotting helpers for the paper figures.

Three functions, used by `simulations.py` and `av_policy_selection.reanalysis`:
  * `build_heat_df`  — long-form dataframe for the elimination heatmap.
  * `heat_plot`      — tile plot of S_t over time (Figure 1a, Figure 3).
  * `make_cs_ribbon_plot` — per-policy [L_t, U_t] ribbons (Figure 1b).
"""

import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    element_text,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_ribbon,
    geom_tile,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_minimal,
)


def build_heat_df(in_set, policy_names, col_indices, t_values):
    """Build a long-form dataframe of policy-elimination status over time.

    Parameters
    ----------
    in_set : ndarray, shape (M, T_full)
        Boolean matrix; ``in_set[i, t]`` is True iff policy i is still in S_t.
    policy_names : list[str], length M
        Display names for each policy row.
    col_indices : iterable[int]
        Column indices into ``in_set`` to subsample (typically every-stride).
    t_values : iterable[int]
        Sample-size labels for the corresponding ``col_indices`` (used on the x-axis).
    """
    rows = []
    for i, name in enumerate(policy_names):
        for j_col, t_val in zip(col_indices, t_values):
            rows.append({
                "t": int(t_val),
                "policy": name,
                "status": "Not eliminated" if in_set[i, j_col] else "Eliminated",
            })
    df = pd.DataFrame(rows)
    # Reverse so the first policy in `policy_names` plots on top.
    df["policy"] = pd.Categorical(df["policy"], categories=policy_names[::-1], ordered=True)
    return df


def heat_plot(df, T_STRIDE, xlim=(0, 2000)):
    """Tile-style heatmap of policy elimination over sample size (Fig. 1a, Fig. 3)."""
    return (
        ggplot(df, aes(x="t", y="policy", fill="status"))
        + geom_tile(aes(width=T_STRIDE), height=0.9)
        + scale_fill_manual(values={"Not eliminated": "#d6604d", "Eliminated": "#aaaaaa"})
        + labs(x="Sample size (t)", y="Policies", fill="")
        + theme_minimal()
        + coord_cartesian(xlim=xlim)
        + theme(legend_position="bottom", title=element_text(hjust=0.5))
    )


def make_cs_ribbon_plot(df_cs, df_nu, xlim=6000, ylim=(0.4, 0.8), hline_size=0.8):
    """Per-policy CS ribbons [L_t, U_t] with dashed true-value lines (Fig. 1b).

    `df_cs` columns: t, policy, lower, upper.
    `df_nu` columns: policy, true_value.
    """
    return (
        ggplot(df_cs, aes(x="t"))
        + geom_ribbon(aes(ymin="lower", ymax="upper"), fill="#d6604d", alpha=0.4)
        + geom_line(aes(y="lower"), color="#d6604d", size=0.3)
        + geom_line(aes(y="upper"), color="#d6604d", size=0.3)
        + geom_hline(
            data=df_nu,
            mapping=aes(yintercept="true_value"),
            linetype="dashed",
            color="black",
            size=hline_size,
        )
        + facet_wrap("policy", ncol=3)
        + coord_cartesian(ylim=ylim, xlim=(0, xlim))
        + labs(x="Sample size (t)", y="Policy value")
        + theme_minimal()
    )
