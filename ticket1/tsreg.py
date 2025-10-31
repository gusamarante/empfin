from empfin import TwoPassReg, ff25p, ff5f

ports = ff25p()
facts, _ = ff5f()

model = TwoPassReg(
    assets=ports,
    factors=facts,
    cs_const=False,
)

print(model.lambdas)

print(model.alphas)