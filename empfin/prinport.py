import numpy as np
import pandas as pd
from numpy.linalg import eigh, svd

# TODO replicate the charts of the paper
# TODO add weight scaling by portfolio target volatility
# TODO add regularization

class PrincipalPortfolios:
    """
    Kelly, Bryan T., Semyon Malamud, and Lasse Heje Pedersen (2023)
    "Principal Portfolios"
    The Journal of Finance, 78(1), 347-387
    https://doi.org/10.1111/jofi.13199
    """

    def __init__(self, returns, signals, signal_transform="rank", pi_weight=None):
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

        pi_weight: None, int, float
            Weighting scheme for the observations of the estimation of the
            predictability matrix Pi. If None, uses equal weights for every
            observation. If any number is passed, the number is used as the
            COM parameter of an exponentially weighting scheme.

        # TODO add attributes
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

        self.Pi = self._estimate_predictability_matrix(pi_weight)
        self.Pi_s, self.Pi_a, U, sv, V, lambdas_s, W, lambdas_a, X, Y = self._decompositions(self.Pi)

        pp_names = [f"PP {k + 1}" for k in range(self.N)]
        pep_names = [f"PEP {k + 1}" for k in range(self.N)]
        pap_names = [f"PAP {k + 1}" for k in range(self.N // 2)]

        # The singular-values / eigenvalues of each strategy are the expected returns
        self.singular_values = pd.Series(sv, index=pp_names, name="Singular Values")
        self.pep_eignvalues = pd.Series(lambdas_s, index=pep_names, name="PEP Eigenvalues")
        self.pap_eignvalues = pd.Series(lambdas_a, index=pap_names, name="PAP Eigenvalues")

        # Portfolio weights
        self.L, self.L_s, self.L_a, self.w, self.w_s, self.w_a = self._portfolio_weights(V, U, lambdas_s, W, X, Y)


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

    def _estimate_predictability_matrix(self, pi_weight):

        if pi_weight is None:  # Equal weights
            Pi = (self.returns.values.T @ self.signals.values) / self.T

        elif isinstance(pi_weight, (float, int)):  # Exponential weights
            assert pi_weight > 0, "`pi_weight` must be non-negative"
            alpha = 1 / (1 + pi_weight)
            decay = (1 - alpha) ** np.arange(self.T - 1, -1, -1)
            w = decay / decay.sum()
            Pi = (self.returns.values * w[:, None]).T @ self.signals.values

        else:
            raise ValueError(f"`pi_weight` needs to be either None or numeric")

        Pi = pd.DataFrame(Pi, index=self.assets, columns=self.assets)
        return Pi

    @staticmethod
    def _decompositions(Pi):

        # Symmetric decomposition of the predictability matrix
        Pi_s = (Pi + Pi.T) / 2
        Pi_a = (Pi - Pi.T) / 2

        # SVD of the predictability matrix
        U, sv, Vt = svd(Pi)
        V = Vt.T

        # Eigen decomposition of the symmetric part of Pi
        lambdas_s, W = eigh(Pi_s)
        order = np.argsort(lambdas_s)[::-1]
        lambdas_s = lambdas_s[order]
        W = W[:, order]

        # Eigen decomposition of the anti-symmetric part of Pi, which
        # keeps only the positive half of the conjugate purely complex
        # eigenvalues. Selecting by position instead of by sign
        # matters when N is odd: the unpaired eigenvalue is zero in theory but
        # can come out as a tiny positive number due to rounding error.
        vals, vecs = np.linalg.eigh(1j * Pi_a)  # ascending order
        n_pairs = Pi_a.shape[0] // 2
        lambdas_a, Wa = vals[::-1][:n_pairs], vecs[:, ::-1][:, :n_pairs]
        X = np.sqrt(2) * Wa.real
        Y = np.sqrt(2) * Wa.imag

        return Pi_s, Pi_a, U, sv, V, lambdas_s, W, lambdas_a, X, Y

    def _portfolio_weights(self, V, U, lambdas_s, W, X, Y):
        # Optimal linear strategy (Proposition 3): L = (Pi'Pi)^(-1/2) Pi' = V U'
        L = V @ U.T

        # Optimal symmetric strategy (eq. 28): L_s = W sign(Lambda_s) W'
        L_s = W @ np.diag(np.sign(lambdas_s)) @ W.T

        # Optimal antisymmetric strategy (Proposition 8): L_a = sum_j (x_j y_j' - y_j x_j')
        L_a = X @ Y.T - Y @ X.T

        # Weights
        w = self.signals @ pd.DataFrame(L, index=self.assets, columns=self.assets)
        w_s = self.signals @ pd.DataFrame(L_s, index=self.assets, columns=self.assets)
        w_a = self.signals @ pd.DataFrame(L_a, index=self.assets, columns=self.assets)

        return L, L_s, L_a, w, w_s, w_a


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
        pi_weight=None,
    )

    print(pp.w)