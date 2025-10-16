import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import colormaps
from plottable import ColDef, Table
from scipy.stats import f
from tqdm import tqdm

from pset1.utils import save_path


show_charts = False
split_date = "1926-12-31"


# ===== Read and Organize the data =====
data = pd.read_csv("data52.csv")
data["Date"] = pd.to_datetime(data["Date"].astype(str).str.replace(".", "/"), format="%Y/%m")
data = data.set_index("Date").resample("QE").last()

# Rolling Sample size
n_sample = (data.index <= split_date).sum()


# ===== Item (a) =====
reg_full = pd.DataFrame()
Y = data["Rm"] - data["Rf"]

for xname in ["dp", "xp"]:
    res = sm.OLS(Y, sm.add_constant(data[xname].shift(1)), missing='drop').fit()
    reg_full.loc["a", xname] = res.params.loc["const"]
    reg_full.loc["b", xname] = res.params.loc[xname]
    reg_full.loc["R2", xname] = res.rsquared

print("ITEM A")
print(reg_full)


# ===== Item (b) =====
dates2loop = data.index[data.index > split_date][:-1]
hist_avg = (data["Rm"] - data["Rf"]).rolling(n_sample, min_periods=n_sample - 1).mean().shift(1)
numer = {"dp": 0, "xp": 0}
denom = {"dp": 0, "xp": 0}
for t in tqdm(dates2loop):
    aux_data = data.loc[:t].iloc[-n_sample:]
    Y = aux_data["Rm"] - aux_data["Rf"]

    for xname in ["dp", "xp"]:
        res = sm.OLS(Y, sm.add_constant(aux_data[xname].shift(1)), missing='drop').fit()
        r_hat = res.predict(exog=[1, aux_data.loc[t, xname]])

        numer[xname] = numer[xname] + (Y.loc[t] - r_hat)[0] ** 2
        denom[xname] = denom[xname] + (Y.loc[t] - hist_avg.loc[t]) ** 2

for xname in ["dp", "xp"]:
    reg_full.loc["R2 OOS", xname] = 1 - numer[xname] / denom[xname]

print("ITEM B")
print(reg_full)
