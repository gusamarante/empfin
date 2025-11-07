from empfin import MacroRiskPremium, ust_futures, us_gdp
import numpy as np


# Grab and transform data
trackers = ust_futures()  # TODO increase sample of bonds
trackers = np.log(trackers.resample("QE").last()).diff(1).dropna()

gdp = us_gdp()
gdp = np.log(gdp).diff(1)

gdp, trackers = gdp.align(trackers, join='inner', axis=0)

mrp = MacroRiskPremium(
    assets=trackers,
    macro_factor=gdp,
    s_bar=8,
)

print(mrp.draws)



