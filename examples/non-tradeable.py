from empfin import NonTradableFactors, us_gdp, us_cpi, ff25p
import pandas as pd


# Non tradable factors
gdp = us_gdp()
cpi = us_cpi()
factors = pd.concat([gdp, cpi], axis=1).resample("QE").last().pct_change().dropna()

# Quarterly returns of the FF 25 portfolios
assets = ff25p(sub_rf=False)
assets = (1 + assets/100).cumprod().resample("QE").last().pct_change(1).dropna()

ntf = NonTradableFactors(assets, factors)
