from empfin import RiskPremiaTermStructure, msb_replication

data_df = msb_replication()
gdp = data_df.pop("GDP")

rpts = RiskPremiaTermStructure(
    assets=data_df,
    factor=gdp,
    s_bar=8,
    n_draws=20,
    burnin=2,
    k=5,
    store_loadings=True,
)

print(rpts.draws_loadings.mean())

# TODO heatmap of the loadings