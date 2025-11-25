from empfin import RiskPremiaTermStructure, bond_futures, us_gdp, plot_correlogram, us_cpi, cds_sov, vix
import numpy as np


# Bonds
# trackers = bond_futures()
# trackers = trackers[trackers.index >= "2001-01-01"].ffill().dropna(axis=1)

# CDS
trackers = cds_sov()
trackers = trackers.drop(
    [
        "Argentina",
        "Greece",
        "Hungary",
        "Ukraine",
        "Russia",
        "Venezuela",
    ],
    axis=1,
)


# With GDP
# gdp = us_gdp()
# gdp = np.log(gdp).diff(1)
# trackers = 100 * np.log(trackers.resample("QE").last()).diff(1).dropna()
# gdp, trackers = gdp.align(trackers, join='inner', axis=0)
# plot_correlogram(gdp)

# With CPI
# cpi = us_cpi()
# cpi = np.log(cpi).diff(1)
# trackers = np.log(trackers.resample("ME").last()).diff(1).dropna()
# cpi, trackers = cpi.align(trackers, join='inner', axis=0)
# plot_correlogram(cpi)

# With VIX
vixi = vix()
vixi = np.log(vixi.resample("QE").mean()).diff(1)
trackers = 100 * np.log(trackers.resample("QE").last()).diff(1).dropna()
vixi, trackers = vixi.align(trackers, join='inner', axis=0)
plot_correlogram(vixi)


mrp = RiskPremiaTermStructure(
    assets=trackers,
    factor=vixi,
    s_bar=4 * 2,
    k=2,
    burnin=100,
    n_draws=1000,
)
mrp.plot_premia_term_structure()
