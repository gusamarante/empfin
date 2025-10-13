import pandas as pd


def performance(rets):
    avg = rets.mean()
    vol = rets.std()
    sharpe = avg / vol
    return pd.DataFrame(
        {
            "Mean": avg,
            "Volatility": vol,
            "Sharpe": sharpe,
        }
    )