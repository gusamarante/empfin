from empfin import RiskPremiaTermStructure, bond_futures, us_gdp, plot_correlogram, us_cpi
import numpy as np
import matplotlib.pyplot as plt


trackers = bond_futures()
trackers = trackers[trackers.index >= "2001-01-01"].ffill().dropna(axis=1)


# With GDP
gdp = us_gdp()
gdp = np.log(gdp).diff(1)
trackers = np.log(trackers.resample("QE").last()).diff(1).dropna()
gdp, trackers = gdp.align(trackers, join='inner', axis=0)
plot_correlogram(gdp)

# With CPI
# cpi = us_cpi()
# cpi = np.log(cpi).diff(1)
# trackers = np.log(trackers.resample("ME").last()).diff(1).dropna()
# cpi, trackers = cpi.align(trackers, join='inner', axis=0)
# plot_correlogram(cpi)


mrp = RiskPremiaTermStructure(
    assets=trackers,
    factor=gdp,
    s_bar=4 * 2,
    # k=2,
    n_draws=1000,
)
print("selected number of factors:", mrp.k)
mrp.plot_premia_term_structure()
