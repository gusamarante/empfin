# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What `empfin` is

A Python package of estimators for empirical asset-pricing models. The shipped artifact is just the `empfin/` package; the repo also carries notebooks (`examples/`) and CSV/XLSX inputs (`sample-data/`) that the notebooks consume but are not installed by `pip`.

Every estimator is a faithful implementation of a specific paper/text — the math and notation track the source, so reach for it when extending or debugging one (full citations in `README.md`):

- `TimeseriesReg`, `CrossSectionReg` — Cochrane (2005), *Asset Pricing*, §12.1 / §12.2.
- `NonTradableFactors` — Campbell, Lo & MacKinlay (1997), *Econometrics of Financial Markets*, §6.2.3 (iterative ML).
- `FamaMacBeth` — Fama & MacBeth (1973), with Newey-West HAC and Shanken (1992) errors-in-variables correction.
- `BFM` / `BFMGLS` / `BFMOMIT` (`bfm.py`) — Bryzgalova, Huang & Julliard, *Bayesian Fama-MacBeth Regressions*.
- `RiskPremiaTermStructure` / `ConditionalRiskPremiaTermStructure` (`msb.py`) — Bryzgalova, Huang & Julliard (2024), *Macro Strikes Back: Term Structure of Risk Premia*; the `msb_*` data readers load the authors' shared replication files.
- `PrincipalPortfolios` / `PrincipalPortfoliosBacktest` — Kelly, Malamud & Pedersen (2023), *Principal Portfolios*. **Under development** — see the `prinport` entry in the module map.

## Common commands

- Install runtime deps: `pip install -r requirements.txt` (same list as `install_requires` in `setup.py`).
- Build locally: `python setup.py sdist bdist_wheel` (matches `.github/workflows/publish.yml`).
- Publish: bump `VERSION` in `setup.py`, then create a GitHub Release — the `publish.yml` workflow builds and uploads to PyPI on release creation via **trusted publishing** (OIDC / `id-token: write` under the `pypi` environment), so there is no PyPI API token in the repo or secrets to look for. There is no other CI.
- Run the notebooks (the de-facto test/validation harness): install `jupyter` separately — it is **not** in `requirements.txt` — then launch from `examples/` so the data readers can resolve `../sample-data/...` paths locally.

There is no test suite, no linter, no type checker.

## Architecture

### Top-level API
`empfin/__init__.py` is the curated public surface. Every estimator/data-reader/chart the user is meant to touch is imported there and listed in `__all__`. When adding a new public symbol, wire it into both lists. Notebooks always import from the top-level `empfin` namespace, not from submodules.

### Estimator conventions (no shared base class)
All estimators follow the same unwritten contract:

- Constructor takes `pandas.DataFrame`s of asset returns and factors with **matching indexes**; any joining/dropping of NaNs happens inside.
- All numerical work happens in `__init__` — there is no separate `.fit()` call. For the Gibbs samplers and Bayesian Fama-MacBeth variants this means construction is the long-running step.
- Results land as instance attributes: point estimates as `lambdas` / `params` / `tstats`; posterior output as `draws_*` DataFrames (e.g. `draws_lambdas`, `draws_betas`, `draws_r2`, `draws_mu_y`, `draws_Sigma_y`).
- Plotting / summary methods on the class operate on those already-stored attributes.

Keep this convention when adding a new estimator — it is what the notebooks and downstream methods assume.

### Module map

- **`empfin/classics.py`** — closed-form / iterative-ML estimators: `TimeseriesReg`, `CrossSectionReg` (composes `TimeseriesReg` internally for the first-pass betas), `FamaMacBeth`, `NonTradableFactors`. The `# TODO models to implement` comment at the top (GMM) is the canonical roadmap for this file. `CrossSectionReg` runs its second stage as either OLS (default) or GLS via `estimator="OLS"/"GLS"` — the two share everything except the second-stage point estimate and its standard errors (Cochrane 2005, 12.9-12.13 for OLS, 12.15-12.17 for GLS). GLS needs `Sigma^-1` (the first-pass residual covariance), so `factors_as_assets=True` — which appends the factor portfolios as test assets, giving them exactly-zero residual variance and forcing the cross-sectional line through them — makes `Sigma` singular; a scale-aware `jitter` diagonal loading (`jitter * max_eigenvalue(Sigma)`) keeps it invertible. `jitter` and the singularity fix are GLS-only; OLS never inverts `Sigma`. `grs_test` has an OLS branch (12.14) and a GLS branch (Shanken-corrected 12.22). `FamaMacBeth` has two operating modes that change its attribute set: full-sample sets `betas` (K × N DataFrame); rolling sets `betas_t` (MultiIndex (date, factor) × asset). Dispatch with `hasattr(self, 'betas')` — see `_average_betas` and `plot_betas_hist` for the pattern. The `'const'` entry in `lambdas` / `lambdas_t` only exists when `cs_const=True`; use `.drop('const', errors='ignore')` and `.get('const', 0)` in helpers that need to work in both modes.
- **`empfin/bfm.py`** — Bayesian Fama-MacBeth family from Bryzgalova-Huang-Julliard. `BFM` is the OLS baseline; `BFMGLS` and `BFMOMIT` subclass it and **only override `_compute_lambdas` / `_compute_r2`** (BFMOMIT also extends `__init__` to take `p`, the number of principal components). New variants should follow the same subclass-and-override pattern instead of duplicating the sampling loop. This file has the package's only cross-module class dependency: `BFM.plot_lambda(include_fm=True)` consumes `FamaMacBeth` from `classics.py` (imported at the top of `bfm.py`). The dependency is one-way — `classics.py` does not import from `bfm.py`, so there is no cycle — but keep it that way if you add code to `classics.py`.
- **`empfin/msb.py`** — "Macro Strikes Back" Gibbs samplers: `RiskPremiaTermStructure` (unconditional) and `ConditionalRiskPremiaTermStructure` (VAR-augmented). The four module-level helpers `_build_V_rho`, `_build_V_eta`, `_build_Sigma_hat`, `_build_Gamma_hat_rho_l` (Newey-West sandwich + regressor construction) are **shared between both classes**. They take a `centered` argument whose semantic meaning differs by caller — centered latent factors in the unconditional sampler, VAR innovations in the conditional one. Preserve that calling convention if you touch them. Downstream methods (`factor_mimicking_portfolio`, `plot_premia_term_structure`, `plot_loadings_heatmap`, …) consume the stored `draws_*`.
- **`empfin/prinport.py` / `empfin/prinport_c.py`** — Principal Portfolios (Kelly, Malamud & Pedersen, 2023). **`PrincipalPortfolios` is still under development**: `prinport.py` is an in-progress rewrite of the class and is not wired into `__init__.py` yet; the public `PrincipalPortfolios`, `PrincipalPortfoliosBacktest` and `momentum_signal` are still imported from `prinport_c.py`. Expect the attributes, signature and file layout to change, and don't treat either file as the settled API. In `prinport.py`, the prediction matrix `Pi` pairs signals and returns from the same row, so callers must lag the signals themselves.
- **`empfin/utils.py`** — `nearest_psd` only. Use it for PSD projection of sampled covariance matrices instead of ad-hoc jitter; it does an eigendecomposition floor and falls back to scale-aware regularization if the projection still has a negative eigenvalue.
- **`empfin/charts.py`** — `plot_correlogram` only. Chart conventions used across the package: aspect ratio `16/7.3`, `tab:blue` primary series with shaded credible/confidence bands, grey 0.5-width gridlines, and the optional `save_path` / `show_chart` parameters seen on the plotting methods inside `msb.py`.

Note: a former monolithic `empfin/factor_models.py` (~2k lines) has been removed in favor of the split above. If you see references to it in older docs or PRs, they predate the reorganization — do not recreate it.

### Data readers (`empfin/data_readers.py`)
Loaders (`ff5f`, `ff25p`, `ust_futures`, `bond_futures`, `us_cpi`, `us_gdp`, `vix`, `msb_replication`, `msb_conditional_replication`) use a **try-local-then-fall-back-to-GitHub** pattern: each reads `../sample-data/<file>` first (works when the user runs notebooks from `examples/` in a clone), and on `FileNotFoundError` re-fetches the same file from the `GITHUB_DATA` raw URL (works for `pip install`ed users who only have the package). When adding a reader, mirror this pattern and commit the file to `sample-data/` so installed users get the fallback. `sample-data/conditional/` is a nested subdirectory consumed by `msb_conditional_replication` — both the local path and the GitHub URL include the `conditional/` subpath; preserve that layout.

## Working with the notebooks
`examples/` contains roughly one notebook per estimator (`timeseries_reg`, `crosssectional_reg`, `fama_macbeth`, `nontradables`, `bayesian_fama_macbeth`, `msb_replication`, `msb_conditional_replication`), plus `cross_sectional_design_choices` (a walkthrough of the OLS-vs-GLS / `cs_const` / `factors_as_assets` design decisions for `CrossSectionReg`). They are integration tests and user documentation in one — when you change a public attribute or constructor signature on an estimator, update the matching notebook. The `msb_*` notebooks run thousands of Gibbs draws; expect minute-scale runtimes.

## Contributing notes (from CONTRIBUTING.md)
Discuss non-trivial changes with the repo owner (issue or email) before implementing. Code should carry comments documenting intentions and edge cases — this is unusual for libraries this small, but the project explicitly asks for it.
