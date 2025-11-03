import pandas as pd


names = pd.read_excel("../data/equity_names.xlsx", index_col="Ticker", sheet_name="Sheet1")

trackers = pd.read_csv("../data/data_equitiy_trackers.csv", index_col=0, sep=";")
trackers.index = pd.to_datetime(trackers.index)

open_int = pd.read_csv("../data/data_equitiy_oi.csv", index_col=0, sep=";")
open_int.index = pd.to_datetime(open_int.index)

results = pd.DataFrame()
for time_freq in ["B", "W-FRI", "ME"]:
    trackers_tf = trackers.resample(time_freq).last()
    open_int_tf = open_int.resample(time_freq).last()

    rets = trackers_tf.pct_change(1)

    autocorr1 = pd.Series({tick: rets[tick].autocorr(lag=1) for tick in rets.columns})
    autocorr2 = pd.Series({tick: rets[tick].autocorr(lag=2) for tick in rets.columns})
    sample = rets.count()

    ew_rets = rets.mean(axis=1)
    autocorr1.loc["EW"] = ew_rets.autocorr(1)
    autocorr2.loc["EW"] = ew_rets.autocorr(2)
    sample.loc["EW"] = ew_rets.count()

    lw_weights = open_int.div(open_int.sum(axis=1), axis=0)
    lw_weights = lw_weights[lw_weights.sum(axis=1) != 0]
    lw_rets = (rets * lw_weights).dropna(how="all").sum(axis=1)
    autocorr1.loc["LW"] = lw_rets.autocorr(1)
    autocorr2.loc["LW"] = lw_rets.autocorr(2)
    sample.loc["LW"] = lw_rets.count()

    results = pd.concat(
        [
            results,
            autocorr1.rename(f"AC1 {time_freq}"),
            autocorr2.rename(f"AC2 {time_freq}"),
            sample.rename(f"Sample Size {time_freq}"),
        ],
        axis=1,
    )

results.to_clipboard()
print(results)