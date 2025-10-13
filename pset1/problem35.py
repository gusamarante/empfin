"""
Problem 3.5 - Campbell's Book
"""
# TODO Checklist
#  Covariance matrix of excess returns
#  compute two ex-post mean-variance efficient sets: one for portfolios not including the riskless asset, and one including the riskless asset. Plot the two sets on a graph with the standard deviation of excess returns on the horizontal axis and the mean excess return on the vertical axis, and indicate where each of the four Fama-French portfolios and the market portfolio lie.
#  Calculate the Sharpe ratios of the tangency portfolio and the market portfolio
import pandas as pd
from empfin import performance
import statsmodels.api as sm


split_date = "1963-12-31"

# Read and Organize the data
data = pd.read_csv("data35.csv")
data = data.rename({"Unnamed: 0": "Date", "Market ": "Market"}, axis=1)
data["Date"] = pd.to_datetime(data["Date"], format="%Y%m")
data = data.set_index("Date").resample("ME").last()
excess = data.subtract(data["Riskfree Rate"], axis=0).drop("Riskfree Rate", axis=1)
subexcess1 = excess[excess.index <= split_date]
subexcess2 = excess[excess.index > split_date]
samples = {
        "Full sample": excess,
        "Sub-sample 1": subexcess1,
        "Sub-sample 2": subexcess2,
}


# Performance Measures
stats_table = pd.concat(
    {
        k: performance(v)
        for k, v in samples.items()
    },
    names=["Sample"],
)

# Alpha and Betas to Market Portfolio
for sample_name, aux_data in samples.items():
    for port in aux_data.columns:
        model = sm.OLS(
            endog=aux_data[port],
            exog=sm.add_constant(aux_data["Market"]),
        ).fit()

        stats_table.loc[(sample_name, port), "Alpha"] = model.params["const"]
        stats_table.loc[(sample_name, port), "t Alpha"] = model.tvalues["const"]
        stats_table.loc[(sample_name, port), "Beta"] = model.params["Market"]
        stats_table.loc[(sample_name, port), "t Beta"] = model.tvalues["Market"]
        stats_table.loc[(sample_name, port), "R-Squared"] = model.rsquared

print(stats_table)


# Correlations
covars = {k: v.cov() for k, v in samples.items()}
correls = {k: v.corr() for k, v in samples.items()}


print(correls)
