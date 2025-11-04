import pandas as pd
import numpy as np


names = pd.read_excel("../data/equity_names.xlsx", index_col="Ticker", sheet_name="Sheet1")

trackers = pd.read_csv("../data/data_equitiy_trackers.csv", index_col=0, sep=";")
trackers.index = pd.to_datetime(trackers.index)

open_int = pd.read_csv("../data/data_equitiy_oi.csv", index_col=0, sep=";")
open_int.index = pd.to_datetime(open_int.index)



results = pd.DataFrame()
for time_freq in ["B", "W-FRI", "ME"]:

    # Frequency-dependent computations
    trackers_tf = trackers.resample(time_freq).last()
    rets = trackers_tf.pct_change(1)

    # Autocorrelations of individual indexes
    autocorr1 = pd.Series({tick: rets[tick].autocorr(lag=1) for tick in rets.columns})
    autocorr2 = pd.Series({tick: rets[tick].autocorr(lag=2) for tick in rets.columns})
    sample = rets.count()

    # Equally-weighted index
    ew_rets = rets.mean(axis=1)
    autocorr1.loc["EW"] = ew_rets.autocorr(1)
    autocorr2.loc["EW"] = ew_rets.autocorr(2)
    sample.loc["EW"] = ew_rets.count()

    # liquidity-weighted index
    lw_weights = open_int.div(open_int.sum(axis=1), axis=0)
    lw_weights = lw_weights[lw_weights.sum(axis=1) != 0]
    lw_rets = (rets * lw_weights).dropna(how="all").sum(axis=1)
    autocorr1.loc["LW"] = lw_rets.autocorr(1)
    autocorr2.loc["LW"] = lw_rets.autocorr(2)
    sample.loc["LW"] = lw_rets.count()

    # Concatenate results
    results = pd.concat(
        [
            results,
            autocorr1.rename(f"AC1 {time_freq}"),
            autocorr2.rename(f"AC2 {time_freq}"),
            sample.rename(f"Sample Size {time_freq}"),
        ],
        axis=1,
    )

# results.to_clipboard()
print(results)


# Conditional on liquidity
open_int_full_mean = open_int.mean()
groups = pd.qcut(open_int_full_mean, 4, labels=False) + 1  # TODO maybe change number o quantiles
autocorr_quantiles = pd.concat(
    [
        results[['AC1 B', 'AC1 W-FRI', 'AC1 ME']].drop(["EW", "LW"], axis=0),
        groups.rename("Group")
    ],
    axis=1,
)
conditional_autocorr = autocorr_quantiles.groupby("Group").mean()
# conditional_autocorr.to_clipboard()
print(conditional_autocorr)

# Cross-correlations (only with weekly)
rets = trackers.resample("W-FRI").last().pct_change(fill_method=None).dropna(how="all").melt(ignore_index=False).dropna()
rets["Group"] = rets["variable"].map(groups)
rets = rets.drop("variable", axis=1).groupby(["date", "Group"]).mean().reset_index().pivot(index="date", columns="Group", values="value")

def cross_correl(df1, df2):
    df1, df2 = df1.dropna().align(df2.dropna(), axis=0, join='inner')
    corr_matrix = np.corrcoef(df1.values.T, df2.values.T)[:df1.shape[1], df1.shape[1]:]
    cross_corr = pd.DataFrame(
        corr_matrix,
        index=df1.columns,
        columns=df2.columns
    )
    return cross_corr

corr_simul = rets.dropna().corr()
corr_cross1 = cross_correl(rets, rets.shift(1))
print(corr_cross1)

# TODO PAREI AQUI -  terminar de montar a matriz de correlações cruzadas
