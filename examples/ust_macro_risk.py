from empfin import MacroRiskPremium, bond_futures, us_gdp, plot_correlogram
import numpy as np


trackers = bond_futures()
trackers = trackers[trackers.index >= "2001-01-01"].ffill().dropna(axis=1)
trackers = np.log(trackers.resample("QE").last()).diff(1).dropna()

gdp = us_gdp()
gdp = np.log(gdp).diff(4)

gdp, trackers = gdp.align(trackers, join='inner', axis=0)

plot_correlogram(gdp)

mrp = MacroRiskPremium(
    assets=trackers,
    macro_factor=gdp,
    s_bar=8,
    # k=1,
    n_draws=100,
)
print("selected number of factors:", mrp.k)
mrp.plot_premia_term_structure()
