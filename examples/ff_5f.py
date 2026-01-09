from empfin import ff25p, ff5f, CrossSectionReg


ports = ff25p()
facts, rf = ff5f()

# ===========================================
# ===== Single Factor Model with CS Reg =====
# ===========================================
cs_reg_mkt_noconst = CrossSectionReg(
    assets=ports,
    factors=facts,
    # factors=facts["Mkt-RF"],
    cs_const=True,
)
print(cs_reg_mkt_noconst.betas.T)
print(cs_reg_mkt_noconst.lambdas)
print(cs_reg_mkt_noconst.grs_test())
