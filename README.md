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

By far the easiest way to replicate the paper figures and interact with the data is to click on the
[Binder badge](https://mybinder.org/v2/gh/dmolitor/av-policy-selection/main?urlpath=rstudio)
in the header of this document. This will bring you to an RStudio IDE instance with all necessary data
and packages installed. Then replicate all figures by executing
```
uv run simulations.py
```
in the terminal.