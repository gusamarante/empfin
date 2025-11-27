from empfin import ff25p, ff5f, TimeseriesReg
import matplotlib.pyplot as plt

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

# --- Chart --- - Model predicted average returns (X) VS historical average, chart the alphas with CIs
# TODO consolidate this chart in the class
size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))
plt.suptitle("Timeseries Regression - Single Factor Model")

# Alphas and their CIs
ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title(r"$\alpha$ and CI")
ax = ts_reg_mkt.params.loc['alpha'].plot(kind='bar', ax=ax, width=0.9)
ax.axhline(0, color="black", lw=0.5)
ax.errorbar(
    ax.get_xticks(),
    ts_reg_mkt.params.loc['alpha'].values,
    yerr=ts_reg_mkt.params_se.loc['alpha'].values * 1.96,
    ls='none',
    ecolor='tab:orange',
)
ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

# Predicted VS actual average returns
ax = plt.subplot2grid((1, 2), (0, 1))

predicted = ts_reg_mkt.params.drop('alpha').multiply(ts_reg_mkt.lambdas, axis=0).sum()
realized = ports.mean()

ax.scatter(predicted, realized, label="Test Assets")
ax.axline((0, 0), (1, 1), color="tab:orange", ls="--", label="45 Degree Line")
ax.set_xlabel(r"Predict Average Monthly Excess Return $\beta_i \lambda_{mkt}$")
ax.set_ylabel(r"Realized Average Monthly Excess Return $E(r_i)$")
ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
ax.legend(frameon=True, loc="upper left")

plt.tight_layout()
plt.show()