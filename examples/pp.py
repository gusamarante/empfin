# TODO this is a temporary file. It is used for testing and it will be erased before merging this branch
from empfin.prinport import PrincipalPortfolios
from empfin import ff25p

# Get returns
rets = ff25p()
rets = rets[rets.index <= "2019-12-31"].dropna()

# Get signals
sigs = rets.rolling(12).sum().shift(1).dropna()

# Align index
new_idx = rets.index.intersection(sigs.index)
rets = rets.reindex(new_idx)
sigs = sigs.reindex(new_idx)

pp = PrincipalPortfolios(
    returns=rets,
    signals=sigs,
    signal_transform="rank",
    pi_weight=None,
)

pp.plot_prediction_matrix()