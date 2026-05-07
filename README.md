# Anytime-valid Optimal Policy Identification — replication package

Code and data to reproduce the four figures in
*"Anytime-valid Optimal Policy Identification"* (Molitor, NeurIPS 2026):

| File                  | Section            | Description                                              |
|-----------------------|--------------------|----------------------------------------------------------|
| `figures/figure1a.png`| 4.1 Illustrative   | Single-run heatmap of the optimal-policy set `S_t`       |
| `figures/figure1b.png`| 4.1 Illustrative   | Per-policy LIL confidence-sequence ribbons               |
| `figures/figure2.png` | 4.2 Sample savings | Mean savings `1 − E[τ]/N₉₀` vs. power-analysis overshoot |
| `figures/figure3.png` | 4.3 Application    | Infodemic re-analysis (Offer-Westort et al., 2024)       |

---

## 1. Install

The project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync
```

This creates a `.venv/` with the locked dependencies (`uv.lock`).

## 2. Get the infodemic data (Figure 3 only)

The raw replication file from the original Nature Human Behaviour paper must be
placed in `data/`:

```bash
# from the project root
curl -L https://raw.githubusercontent.com/gsbDBI/infodemic-replication/main/data/cleaned-data_2023-03-28.csv \
     -o data/cleaned-data_2023-03-28.csv
```

(If the URL changes, the file lives at `data/cleaned-data_2023-03-28.csv` in
<https://github.com/gsbDBI/infodemic-replication>.)

If this file is missing, Figures 1, 2 still reproduce; Figure 3 is skipped with
a warning.

## 3. Reproduce all four figures

Two modes:

```bash
# Quick smoke run (~1 minute, smaller grids and fewer trials).
uv run python simulations.py

# Full paper settings (longer; uses joblib parallelism across all cores).
FULL_RUN=1 uv run python simulations.py
```

All output figures land in `figures/`. Intermediate CSVs from Figure 2's grid
are saved to `data/fig2_savings.csv`.

## 4. (Optional) Run the test suite

```bash
uv run pytest -q
```

The suite covers the LIL, Betting, and PrPL confidence-sequence formulas, the
elimination/stopping-time logic in `PolicySelector`, and the OLS reward
predictor used to form the AIPW pseudo-outcomes.

---

## Repository layout

```
av-policy-selection/
├── simulations.py                 # entry-point — produces all four figures
├── pyproject.toml                 # uv-managed dependencies
├── uv.lock
├── data/
│   └── cleaned-data_2023-03-28.csv  # infodemic raw data (downloaded by user)
├── figures/                         # output directory (created at runtime)
├── src/
│   ├── utils.py                     # plotting helpers (heatmap, CS ribbons)
│   └── av_policy_selection/
│       ├── __init__.py              # public API
│       ├── confidence_sequences.py  # LIL / PrPL / Betting CSs
│       ├── policy_selection.py      # PolicySelector (S_t and stopping time τ)
│       ├── reward_predictors.py     # OLS reward model for AIPW
│       ├── load_data.py             # infodemic data loader + arm/policy maps
│       └── reanalysis.py            # infodemic pipeline (Figure 3)
└── tests/                           # pytest suite for the library code
```

## What `simulations.py` does

1. **Figure 1 (illustrative example).** Generates one synthetic trial with five
   candidate policies plus the logging policy (six total), runs the LIL
   confidence sequence with Bonferroni correction at `α/(M+1)`, and saves the
   `S_t` heatmap (`figure1a.png`) and per-policy CS ribbons (`figure1b.png`).

2. **Figure 2 (power-analysis overshoot).** For each powered-for gap
   `Δ_powered ∈ {0.02, 0.05, 0.10, 0.15, 0.20}`, finds the oracle fixed sample
   size `N₉₀` (smallest `N` at which the PrPL CI of Luedtke & Soni (2024)
   identifies the optimal policy with probability ≥ 0.90) by binary search,
   then runs the anytime-valid PrPL confidence sequence at true gaps
   `Δ_true = c · Δ_powered` for `c ∈ {1.25, 1.5, 1.75, 2, 2.5, 3}` with a
   budget cap of `5·N₉₀`. The plot averages `1 − E[τ]/N₉₀` across powered-for
   gaps for each `c`.

3. **Figure 3 (infodemic).** Loads the Offer-Westort et al. (2024) data,
   forms AIPW pseudo-outcomes for the eight constant respondent-level
   policies using an OLS reward predictor over all 40 arms, runs the betting
   confidence sequence at level `α/8`, and saves the elimination heatmap.

## Citation

If this code is useful, please cite the paper. The infodemic analysis builds on
data released by the original authors at
<https://github.com/gsbDBI/infodemic-replication>.
