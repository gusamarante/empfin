from empfin import ff25p, ff5f, TimeseriesReg

ports = ff25p()
facts, rf = ff5f()

# Single Factor Model - Market
ts_reg_mkt = TimeseriesReg(
    assets=ports,
    factors=facts["Mkt-RF"],
)

print(ts_reg_mkt.tstats)