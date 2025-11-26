import pandas as pd
from empfin import RiskPremiaTermStructure, plot_correlogram

data_df = pd.read_csv("../sample-data/GDP_data_clean.csv", index_col=0)
data_df.index = pd.to_datetime(data_df.index)

gdp = data_df.pop("GDP")
plot_correlogram(gdp)

pf = RiskPremiaTermStructure(
    assets=data_df,
    factor=gdp,
    s_bar=12,
    n_draws=2000,
    burnin=2000,
    k=5,
)
pf.plot_premia_term_structure()
