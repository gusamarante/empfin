from empfin import TwoPassReg, ff25p, ff5f

ports = ff25p()
facts, _ = ff5f()

model = TwoPassReg(
    assets=ports,
    factors=facts,
    cs_const=True,
)

print(model.lambdas)
print(model.alphas)
print(model.shanken_factor)
print(model.conv_cov_alpha_hat)
print(model.shanken_cov_alpha_hat)
