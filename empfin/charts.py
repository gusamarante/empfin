import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from matplotlib.ticker import MultipleLocator


def plot_correlogram(timeseries, size=6):
    # TODO documetation
    # TODO nlags
    correlogram = pd.DataFrame(
        {
            "AC": acf(timeseries, nlags=20)[1:],
            "PAC": pacf(timeseries, nlags=20)[1:],
        },
        index=range(1, 21)
    )
    fig = plt.figure(figsize=(size * (16 / 7.3), size))
    ax = plt.subplot2grid((2, 1), (0, 0))
    rects = ax.bar(correlogram["AC"].index, correlogram["AC"].values)
    ax.bar_label(rects, padding=1, fmt="{:.3f}")
    ax.axhline(0, color='black', lw=0.5)
    ax.set(ylim=(-1, 1), title="AC Function")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

    ax = plt.subplot2grid((2, 1), (1, 0))
    rects = ax.bar(correlogram["PAC"].index, correlogram["PAC"].values)
    ax.bar_label(rects, padding=1, fmt="{:.3f}")
    ax.axhline(0, color='black', lw=0.5)
    ax.set(ylim=(-1, 1), title="PAC Function")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    # TODO save figure
    plt.show()