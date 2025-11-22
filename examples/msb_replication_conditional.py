import pandas as pd
import numpy as np
from pathlib import Path
from empfin import PersistentFactors

data_path = Path(r"../sample-data")

# === RETURNS ===
assets = pd.read_csv(data_path.joinpath("GDP_data_clean.csv"), index_col=0)
assets.index = pd.to_datetime(assets.index)
assets = assets.drop("GDP", axis=1)


# === READ MACRO DATA ===
# GDP
gdp = pd.read_excel(
    data_path.joinpath("GDPC1.xlsx"),
    sheet_name="Quarterly",
    index_col=0,
)
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.resample("QE").last()
gdp = gdp["GDPC1"].rename("GDP")


# Industrial Production
indpro = pd.read_excel(
    data_path.joinpath("INDPRO.xlsx"),
    sheet_name="Monthly",
    index_col=0,
)
indpro.index = pd.to_datetime(indpro.index)
indpro = indpro.resample("QE").last()
indpro = indpro["INDPRO"].rename("IndProd")


# Durable Consumption
durc = pd.read_excel(
    data_path.joinpath("PCEDG.xlsx"),
    sheet_name="Monthly",
    index_col=0,
)
durc.index = pd.to_datetime(durc.index)
durc = durc.resample("QE").last()
durc = durc["PCEDG"].rename("DurC")


# Non-Durable Consumption
ndurc = pd.read_excel(
    data_path.joinpath("PCEND.xlsx"),
    sheet_name="Monthly",
    index_col=0,
)
ndurc.index = pd.to_datetime(ndurc.index)
ndurc = ndurc.resample("QE").last()
ndurc = ndurc["PCEND"].rename("NonDurC")


# Service Consumption
servc = pd.read_excel(
    data_path.joinpath("PCES.xlsx"),
    sheet_name="Monthly",
    index_col=0,
)
servc.index = pd.to_datetime(servc.index)
servc = servc.resample("QE").last()
servc = servc["PCES"].rename("ServC")


# All together now...
pred_vars = pd.concat([gdp, indpro, durc, ndurc], axis=1)
pred_vars = np.log(pred_vars).diff(1)

start_date, end_date = assets.index.min(), assets.index.max()
pred_vars = pred_vars[pred_vars.index >= start_date]
pred_vars = pred_vars[pred_vars.index <= end_date]

pf = PersistentFactors(
    assets=assets,
    macro_factor=pred_vars["GDP"],
    s_bar=12,
    n_draws=200,
    burnin=10,
    k=5,
    cond_vars=pred_vars,
)
pf.plot_premia_term_structure()
