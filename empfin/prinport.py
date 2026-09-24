

class PrincipalPortfolios:
    """
    Kelly, Bryan T., Semyon Malamud, and Lasse Heje Pedersen (2023)
    "Principal Portfolios"
    The Journal of Finance, 78(1), 347-387
    https://doi.org/10.1111/jofi.13199
    """

    def __init__(self, returns, signals, signal_transform="rank", pi_weights=None):
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

        pi_weights: str, float
            Weighting scheme for the observations of the estimation of the
            predictability matrix Pi. If None, uses equal weights for every
            observation. If any number is passed, the number is used as the
            COM parameter of an exponentially weighting scheme
        """

        assert returns.index.equals(signals.index), \
            "Indexes of `assets` and `factors` must be the same"

        assert returns.columns.equals(signals.columns), \
            "Columns of `assets` and `factors` must be the same"

        self.signals = self._transform_signals(signals, signal_transform)
        self.signal_transform = signal_transform
        self.returns = returns
        self.T, self.N = self.returns.shape
        self.assets = self.returns.columns

        self.Pi = self._estimate_predictability_matrix(pi_weights)

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

    def _estimate_predictability_matrix(self, pi_weights):
        # TODO parei aqui, estimar a matriz PI com os dois estimadores
        #  disponíves
        pass

if __name__ == "__main__":
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
    )
    print(pp.signals)