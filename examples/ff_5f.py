from empfin import ff25p, ff5f, CrossSectionReg


ports = ff25p()
facts, rf = ff5f()

# ===========================================
# ===== Single Factor Model with CS Reg =====
# ===========================================
cs_reg_mkt_noconst = CrossSectionReg(
    assets=ports,
    factors=facts,
    cs_const=False,
)