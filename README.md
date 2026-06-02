# Anytime-valid Optimal Policy Identification

<!-- badges: start -->
[![Launch RStudio Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dmolitor/av-policy-selection/main?urlpath=rstudio)
<!-- badges: end -->

Replication materials for [Anytime-valid Optimal Policy Identification (Molitor, 2026).](https://www.dmolitor.com/blog/posts/optimal_policy_id/)

## Code and data description

All data required to replicate figures for this paper as well as intermediate data outputs
will be stored in the `/data` directory. Corresponding code can be found in the `src/` directory.
All figures will be stored in the `figures/` directory.

## Replicating figures - Binder

The easiest way to replicate the paper figures and interact with the data is to click on the
[Binder badge](https://mybinder.org/v2/gh/dmolitor/av-policy-selection/main?urlpath=rstudio)
in the header of this document. This will bring you to an RStudio instance with all necessary data
and packages installed. Then replicate all figures by executing
```
uv run simulations.py
```
in the terminal.

## Replicating figures - local

### Install packages (with pinned versions)

To install the required packages with specific versions used in the analysis,
first [install uv](https://docs.astral.sh/uv/getting-started/installation/) and sync the local project:
```r
uv sync
```

### Replicating figures

Once packages have been installed, replicate the figures with the following:
```
uv run simulations.py
```

## Docker image

A Dockerfile is provided for a Docker image with Python and all necessary packages installed.

## Table of contents
```
.
├── data                                # Directory for raw data and intermediate outputs
│   └── cleaned-data_2023-03-28.csv     # Raw data to replicate Offer-Westort et al. misinformation study
├── Dockerfile                          # Dockerfile for building a Docker image for project
├── figures                             # Directory for output figures
├── pyproject.toml
├── README.md
├── simulations.py                      # Primary script for replicating figures
├── src                                 # Directory containing all code for replicating figures
│   ├── av_policy_selection
│   │   ├── __init__.py
│   │   ├── confidence_sequences.py
│   │   ├── load_data.py
│   │   ├── policy_selection.py
│   │   ├── py.typed
│   │   ├── reanalysis.py
│   │   └── reward_predictors.py
│   └── utils.py
└── uv.lock                             # Lockfile for uv to install dependencies from
```