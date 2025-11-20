import pandas as pd
from empfin import PersistentFactors

data_df = pd.read_csv("../sample-data/GDP_data_clean.csv", index_col=0)
data_df.index = pd.to_datetime(data_df.index)

gdp = data_df.pop("GDP")

pf = PersistentFactors(
    assets=data_df,
    macro_factor=gdp,
    s_bar=12,
    n_draws=2000,
    burnin=2000,
    k=5,  # TODO let this loose
)
pf.plot_premia_term_structure()

