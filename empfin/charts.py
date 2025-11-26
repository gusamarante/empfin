import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from matplotlib.ticker import MultipleLocator


def plot_correlogram(timeseries, size=6, nlags=20):
    """
    Plot the autocorrelation and partial autocorrelation functions

    Parameters
    ----------
    timeseries: pd.Series
        Timeseries of interest

    size: float
        Relative size of the chart. The aspect ratio is fixed at (16 / 7.3)

    nlags: int
        Number of lags to plot for the ACF and PACF
    """
    correlogram = pd.DataFrame(
        {
            "AC": acf(timeseries, nlags=nlags)[1:],
            "PAC": pacf(timeseries, nlags=nlags)[1:],
        },
        index=range(1, nlags + 1)
    )
    plt.figure(figsize=(size * (16 / 7.3), size))
    ax = plt.subplot2grid((2, 1), (0, 0))
    rects = ax.bar(correlogram["AC"].index, correlogram["AC"].values)
    ax.bar_label(rects, padding=1, fmt="{:.3f}")
    ax.axhline(0, color='black', lw=0.5)
    ax.set(xlim=(0.5, None), ylim=(-1, 1), title="AC Function")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

    ax = plt.subplot2grid((2, 1), (1, 0))
    rects = ax.bar(correlogram["PAC"].index, correlogram["PAC"].values)
    ax.bar_label(rects, padding=1, fmt="{:.3f}")
    ax.axhline(0, color='black', lw=0.5)
    ax.set(xlim=(0.5, None), ylim=(-1, 1), title="PAC Function")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()