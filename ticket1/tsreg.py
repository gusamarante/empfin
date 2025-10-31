from empfin import TwoPassOLS, ff25p, ff5f

ports = ff25p()
facts, _ = ff5f()

model = TwoPassOLS(
    assets=ports,
    factors=facts,
    cs_const=True,
)

