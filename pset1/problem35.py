"""
Problem 3.5 - Campbell's Book
"""
# TODO Checklist
#  compute two ex-post mean-variance efficient sets: one for portfolios not including the riskless asset, and one including the riskless asset. Plot the two sets on a graph with the standard deviation of excess returns on the horizontal axis and the mean excess return on the vertical axis, and indicate where each of the four Fama-French portfolios and the market portfolio lie.
#  Calculate the Sharpe ratios of the tangency portfolio and the market portfolio
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import colormaps
from plottable import ColDef, Table
import numpy as np


split_date = "1963-12-31"


# ===== Custom Cunctions =====
def performance(rets):
    avg = rets.mean()
    vol = rets.std()
    sharpe = avg / vol
    return pd.DataFrame(
        {
            "Mean": avg,
            "Volatility": vol,
            "Sharpe": sharpe,
        }
    )

def plot_efficient_frontier(axis, mu, sigma):
    inv_sigma = np.linalg.inv(sigma)
    ones = np.ones(len(mu))

    w_gmv = (inv_sigma @ ones) / (ones @ inv_sigma @ ones)
    mu_gmv = w_gmv @ mu
    vol_gmv = np.sqrt(w_gmv @ sigma @ w_gmv)

    w_tan = (inv_sigma @ mu) / (ones @ inv_sigma @ mu)
    mu_tan = w_tan @ mu
    vol_tan = np.sqrt(w_tan @ sigma @ w_tan)

    for port in mu.index:
        axis.plot(np.sqrt(sigma.loc[port, port]), mu.loc[port], label=port, marker='o', lw=0)

    axis.plot(vol_tan, mu_tan, label="Tangency", marker='s', lw=0)
    axis.plot(vol_gmv, mu_gmv, label="GMV", marker='s', lw=0)

    # Efficient Frontier
    A = mu @ inv_sigma @ mu
    B = ones @ inv_sigma @ mu
    C = ones @ inv_sigma @ ones

    def min_risk(m):
        return np.sqrt((C * (m ** 2) - 2 * B * m + A) / (A * C - B ** 2))

    max_mu = mu_tan + 1
    mu_range = np.arange(mu_gmv, max_mu, (max_mu - mu_gmv) / 100)
    sigma_range = np.array(list(map(min_risk, mu_range)))

    axis.plot(sigma_range, mu_range, label="EF Risky")
    axis.axline((0, 0), (vol_tan, mu_tan), label="EF Risky + Rf")

    return axis


# ===== Read and Organize the data =====
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


# ===== Performance Measures =====
stats_table = pd.concat(
    {
        k: performance(v)
        for k, v in samples.items()
    },
    names=["Sample"],
)

# ===== Alpha and Betas to Market Portfolio =====
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


# ===== Correlations =====
covars = {k: v.cov() for k, v in samples.items()}
correls = {k: v.corr() for k, v in samples.items()}

column_definitions = [
    ColDef(port, cmap=colormaps['coolwarm'], formatter="{:.2f}") for port in subexcess1.columns
] + [ColDef("index", title="")]
cellkw = {"edgecolor": "w", "linewidth": 0}


size = 6
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((2, 4), (0, 1), colspan=2)
ax.set_title("Full Sample")
tab = Table(correls['Full sample'], column_definitions=column_definitions, cell_kw=cellkw)

ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title("Sub-Sample 1")
tab = Table(correls['Sub-sample 1'], column_definitions=column_definitions, cell_kw=cellkw)

ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("Sub-Sample 2")
tab = Table(correls['Sub-sample 2'], column_definitions=column_definitions, cell_kw=cellkw)

plt.tight_layout()
plt.show()


# ===== Efficient Frontioers =====
size = 8
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((2, 4), (0, 1), colspan=2)
ax.set_title("Full Sample")
ax = plot_efficient_frontier(
    axis=ax,
    mu=stats_table.loc["Full sample", "Mean"],
    sigma=covars["Full sample"],
)
ax.legend(frameon=True, loc='best')

ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title("Sub-Sample 1")
ax = plot_efficient_frontier(
    axis=ax,
    mu=stats_table.loc["Sub-sample 1", "Mean"],
    sigma=covars["Sub-sample 1"],
)
ax.legend(frameon=True, loc='best')

ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("Sub-Sample 2")
ax = plot_efficient_frontier(
    axis=ax,
    mu=stats_table.loc["Sub-sample 2", "Mean"],
    sigma=covars["Sub-sample 2"],
)
ax.legend(frameon=True, loc='best')

plt.tight_layout()
plt.show()
