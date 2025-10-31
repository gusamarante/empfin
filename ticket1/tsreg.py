from empfin import TimeseriesReg, TwoPassOLS, ff25p, ff5f
from bayesfm import BFMOMIT, BFMGLS


# Read data
ports = ff25p()
facts, _ = ff5f()


# Timeseries OLS Regression
model_ts = TimeseriesReg(
    assets=ports,
    factors=facts,
)
print(model_ts.lambdas)



# Cross-sectional regression / 2-pass OLS
model_cs = TwoPassOLS(
    assets=ports,
    factors=facts,
    cs_const=False,
)
print(model_cs.lambdas)


# Bayesian Fama-MacBeth
bfm = BFMGLS(
    assets=ports,
    factors=facts,
    n_draws=100_000,
    # p=10,
)
print(bfm.ci_table_lambda())
bfm.plot_lambda(include_fm=True)
