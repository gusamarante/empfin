import pandas as pd
from empfin import RiskPremiaTermStructure

data_df = pd.read_csv("../sample-data/GDP_data_clean.csv", index_col=0)
data_df.index = pd.to_datetime(data_df.index)

gdp = data_df.pop("GDP")

pf = RiskPremiaTermStructure(
    assets=data_df,
    macro_factor=gdp,
    s_bar=12,
    n_draws=100,
    burnin=10,
    k=5,
)
pf.plot_premia_term_structure()
