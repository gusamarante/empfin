from empfin import ff25p, ff5f, TimeseriesReg

# TODO make a chart for the lambdas, similar to the alphas

ports = ff25p()
facts, rf = ff5f()

# ===========================================
# ===== Single Factor Model with TS Reg =====
# ===========================================
ts_reg_mkt = TimeseriesReg(
    assets=ports,
    factors=facts["Mkt-RF"],
)

# --- GRS test ---
print(ts_reg_mkt.grs_test())
ts_reg_mkt.plot_alpha_pred(title="Timeseries Regression - Single Factor Model")


# ======================================
# ===== 5 Factor Model with TS Reg =====
# ======================================
ts_reg_5f = TimeseriesReg(
    assets=ports,
    factors=facts,
)

# --- GRS test ---
print(ts_reg_5f.grs_test())
ts_reg_5f.plot_alpha_pred(title="Timeseries Regression - 5 Factor Model")
