

class PrincipalPortfolios:
    """
    Kelly, Bryan T., Semyon Malamud, and Lasse Heje Pedersen (2023)
    "Principal Portfolios"
    The Journal of Finance, 78(1), 347-387
    https://doi.org/10.1111/jofi.13199
    """

    def __init__(self, returns, signals, signal_transform="rank"):
        """
        Full-sample (in-sample) estimation of the principal portfolios. The
        framework uses the signals of all assets to predict the return of each
        individual asset, including cross-predictability.

        Parameters
        ----------
        returns: pandas.DataFrame
            Timeseries of asset (excess) returns

        signals: pandas.DataFrame
            Timeseries of the predictive signals. Must have the same columns
            as `returns`. The signal in the row of date t is assumed to be
            known at date t and is paired with the return of date t + 1, so
            both panels should be indexed by the date at which they are
            observed. Dates with any missing value are dropped.

        signal_transform: str
            Cross-sectional transformation applied to the signals at each
            date before pairing. "rank" (default, as in the paper) maps the
            signals to equally spaced values in [-0.5, 0.5], making them
            dollar-neutral and insensitive to outliers. "zscore"
            standardizes each cross-section. "none" uses the raw signals.
        """

        assert returns.index.equals(signals.index), \
            "Indexes of `assets` and `factors` must be the same"

        assert returns.columns.equals(signals.columns), \
            "Columns of `assets` and `factors` must be the same"

        self.signals = self._transform_signals(signals, signal_transform)
        self.signal_transform = signal_transform

    @staticmethod
    def _transform_signals(signals, signal_transform):
        """
        Applies the cross-sectional signal transformation. "rank" maps signals
        at each date to equally spaced values in the interval [-0.5, 0.5],
        which makes the signal vector dollar-neutral and robust to outliers.
        "zscore" standardizes each cross-section, and "none" leaves the raw
        signals untouched.
        """
        if signal_transform == "rank":
            ranks = signals.rank(axis=1)
            n_valid = signals.count(axis=1)
            return ranks.div(n_valid, axis=0).sub((n_valid + 1) / (2 * n_valid), axis=0)

        elif signal_transform == "zscore":
            mu = signals.mean(axis=1)
            sd = signals.std(axis=1)
            return signals.sub(mu, axis=0).div(sd, axis=0)

        elif signal_transform == "none":
            return signals.copy()

        else:
            raise ValueError(
                f"`signal_transform` must be 'rank', 'zscore' or 'none'. Got {signal_transform!r}"
            )

if __name__ == "__main__":
    # TODO read data and test it
    pass