"""
Problem 3.5 - Campbell's Book
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import colormaps
from plottable import ColDef, Table
from scipy.stats import f

from pset1.utils import save_path


show_charts = False
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

    return axis, w_tan


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
plt.savefig(save_path.joinpath("Q1 Correlations.pdf"))
if show_charts:
    plt.show()
plt.close()


# ===== Efficient Frontiers =====
size = 8
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((2, 4), (0, 1), colspan=2)
ax.set_title("Full Sample")
ax, w_tan_full = plot_efficient_frontier(
    axis=ax,
    mu=stats_table.loc["Full sample", "Mean"],
    sigma=covars["Full sample"],
)
ax.legend(frameon=True, loc='best')

ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title("Sub-Sample 1")
ax, w_tan_ss1 = plot_efficient_frontier(
    axis=ax,
    mu=stats_table.loc["Sub-sample 1", "Mean"],
    sigma=covars["Sub-sample 1"],
)
ax.legend(frameon=True, loc='best')

ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("Sub-Sample 2")
ax, w_tan_ss2 = plot_efficient_frontier(
    axis=ax,
    mu=stats_table.loc["Sub-sample 2", "Mean"],
    sigma=covars["Sub-sample 2"],
)
ax.legend(frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath("Q1 Efficient Frontiers.pdf"))
if show_charts:
    plt.show()
plt.close()


# Tangency portfolios
rets_tan_full = (excess @ w_tan_full).to_frame("Tangency")
rets_tan_ss1 = (subexcess1 @ w_tan_ss1).to_frame("Tangency")
rets_tan_ss2 = (subexcess2 @ w_tan_ss2).to_frame("Tangency")

sharpes = pd.DataFrame()
sharpes.loc["Full sample", "Market"] = stats_table.loc[("Full sample", "Market"), "Sharpe"]
sharpes.loc["Full sample", "Tangency"] = performance(rets_tan_full).loc["Tangency", "Sharpe"]
sharpes.loc["Full sample", "T"] = rets_tan_full.shape[0]
sharpes.loc["Sub-sample 1", "Market"] = stats_table.loc[("Sub-sample 1", "Market"), "Sharpe"]
sharpes.loc["Sub-sample 1", "Tangency"] = performance(rets_tan_ss1).loc["Tangency", "Sharpe"]
sharpes.loc["Sub-sample 1", "T"] = rets_tan_ss1.shape[0]
sharpes.loc["Sub-sample 2", "Market"] = stats_table.loc[("Sub-sample 2", "Market"), "Sharpe"]
sharpes.loc["Sub-sample 2", "Tangency"] = performance(rets_tan_ss2).loc["Tangency", "Sharpe"]
sharpes.loc["Sub-sample 2", "T"] = rets_tan_ss2.shape[0]

sharpes["GRS"] = ((sharpes["Tangency"]**2 - sharpes["Market"]**2) / (1 + sharpes["Market"]**2)) * ((sharpes["T"] - 5 - 1) / 5)
sharpes["pvalue"] = 1 - f.cdf(sharpes["GRS"], dfn=5, dfd=sharpes["T"] - 5 - 1)

print(sharpes)


# ===== Chart - Beta =====
size = 8
fig = plt.figure(figsize=(size * (16 / 9), size))

aux_plot = stats_table.loc["Full sample", ["Mean", "Beta"]].copy()
ax = plt.subplot2grid((2, 4), (0, 1), colspan=2)
ax.set_title("Full Sample")
for port in aux_plot.index:
    ax.plot(aux_plot.loc[port, "Beta"], aux_plot.loc[port, "Mean"], marker='o', lw=0, label=port)
ax.axline((0, 0), (aux_plot.loc["Market", "Beta"], aux_plot.loc["Market", "Mean"]), label=None)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Average Return")
ax.legend(frameon=True, loc='best')


aux_plot = stats_table.loc["Sub-sample 1", ["Mean", "Beta"]].copy()
ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title("Sub-Sample 1")
for port in aux_plot.index:
    ax.plot(aux_plot.loc[port, "Beta"], aux_plot.loc[port, "Mean"], marker='o', lw=0, label=port)
ax.axline((0, 0), (aux_plot.loc["Market", "Beta"], aux_plot.loc["Market", "Mean"]), label=None)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Average Return")
ax.legend(frameon=True, loc='best')


aux_plot = stats_table.loc["Sub-sample 2", ["Mean", "Beta"]].copy()
ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("Sub-Sample 2")
for port in aux_plot.index:
    ax.plot(aux_plot.loc[port, "Beta"], aux_plot.loc[port, "Mean"], marker='o', lw=0, label=port)
ax.axline((0, 0), (aux_plot.loc["Market", "Beta"], aux_plot.loc["Market", "Mean"]), label=None)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Average Return")
ax.legend(frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath("Q1 Beta Return SML.pdf"))
if show_charts:
    plt.show()
plt.close()


# ==================
# ===== ITEM B =====
# ==================
smallHML = excess["Small-High"] - excess["Small-Low"]
smallHML1yMA = smallHML.rolling(12).mean().dropna()
smallHML_ss1 = smallHML[smallHML.index <= split_date]
smallHML_ss2 = smallHML[smallHML.index > split_date]

perf_smallHML = pd.DataFrame(
    {
        "Full Sample": performance(smallHML.to_frame("Full Sample")).loc["Full Sample"],
        "Sub-sample 1": performance(smallHML_ss1.to_frame("Sub-sample 1")).loc["Sub-sample 1"],
        "Sub-sample 2": performance(smallHML_ss2.to_frame("Sub-sample 2")).loc["Sub-sample 2"],
    }
)


# ===== Chart - Beta =====
size = 6
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(smallHML, color="tab:blue", alpha=0.3, label="Small HML")
ax.plot(smallHML1yMA, color="tab:blue", label="Small HML 1y MA")


ax.plot(
    [smallHML_ss1.index[0], smallHML_ss1.index[-1]],
    [perf_smallHML.loc["Mean", "Sub-sample 1"], perf_smallHML.loc["Mean", "Sub-sample 1"]],
    label="Sub-sample 1 mean",
    color="tab:orange",
)
ax.fill_between(
    smallHML_ss1.index,
    perf_smallHML.loc["Mean", "Sub-sample 1"] + 2 * perf_smallHML.loc["Volatility", "Sub-sample 1"],
    perf_smallHML.loc["Mean", "Sub-sample 1"] - 2 * perf_smallHML.loc["Volatility", "Sub-sample 1"],
    label="Sub-sample 1 $\pm2\sigma$",
    color="tab:orange",
    alpha=0.15,
)


ax.plot(
    [smallHML_ss2.index[0], smallHML_ss2.index[-1]],
    [perf_smallHML.loc["Mean", "Sub-sample 2"], perf_smallHML.loc["Mean", "Sub-sample 2"]],
    label="Sub-sample 2 mean",
    color="tab:red",
)
ax.fill_between(
    smallHML_ss2.index,
    perf_smallHML.loc["Mean", "Sub-sample 2"] + 2 * perf_smallHML.loc["Volatility", "Sub-sample 2"],
    perf_smallHML.loc["Mean", "Sub-sample 2"] - 2 * perf_smallHML.loc["Volatility", "Sub-sample 2"],
    label="Sub-sample 2 $\pm2\sigma$",
    color="tab:red",
    alpha=0.15,
)

ax.axvline(pd.to_datetime(split_date), color="black", ls="--", label="Sample Split")
ax.legend(frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath("Q1 SmallHML timeseries.pdf"))
if show_charts:
    plt.show()
plt.close()


# TODO Item (b) (iii) Auto correlation

