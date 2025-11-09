from empfin import MacroRiskPremium, bond_futures, us_gdp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from matplotlib.ticker import MultipleLocator



# Grab and transform data
trackers = bond_futures()
trackers = trackers[trackers.index >= "2001-01-01"].ffill().dropna(axis=1)
trackers = np.log(trackers.resample("QE").last()).diff(1).dropna()

gdp = us_gdp()
gdp = np.log(gdp).diff(4)

gdp, trackers = gdp.align(trackers, join='inner', axis=0)

# TODO turn this into a function
correlogram = pd.DataFrame(
    {
        "AC": acf(gdp, nlags=20)[1:],
        "PAC": pacf(gdp, nlags=20)[1:],
    },
    index=range(1, 21)
)

size = 6
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((2, 1), (0, 0))
rects = ax.bar(correlogram["AC"].index, correlogram["AC"].values)
ax.bar_label(rects, padding=1, fmt="{:.3f}")
ax.axhline(0, color='black', lw=0.5)
ax.set(ylim=(-1, 1), title="AC Function")
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

ax = plt.subplot2grid((2, 1), (1, 0))
rects = ax.bar(correlogram["PAC"].index, correlogram["PAC"].values)
ax.bar_label(rects, padding=1, fmt="{:.3f}")
ax.axhline(0, color='black', lw=0.5)
ax.set(ylim=(-1, 1), title="PAC Function")
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.show()


mrp = MacroRiskPremium(
    assets=trackers,
    macro_factor=gdp,
    s_bar=8,
    # k=1,
    n_draws=1000,
)
print("selected number of factors:", mrp.k)
mrp.plot_premia_term_structure()

