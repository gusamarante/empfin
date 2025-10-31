from empfin import TimeseriesReg, ff25p, ff5f

ports = ff25p()
facts, _ = ff5f()

model = TimeseriesReg(
    assets=ports,
    factors=facts,
)

print(model.params)