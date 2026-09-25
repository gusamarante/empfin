import numpy as np
import pandas as pd
from numpy.linalg import eigh, svd
import matplotlib.pyplot as plt
import seaborn as sns

class PrincipalPortfolios:
    """
    Kelly, Bryan T., Semyon Malamud, and Lasse Heje Pedersen (2023)
    "Principal Portfolios"
    The Journal of Finance, 78(1), 347-387
    https://doi.org/10.1111/jofi.13199
    """

    def __init__(
            self,
            returns,
            signals,
            signal_transform="rank",
            pi_weight=None,
            rank=None,
            p_norm=np.inf,
    ):
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

        rank: None, int
            Regularization parameter K of Proposition 11: the optimal
            strategies are restricted to position matrices with rank(L) <= K,
            which keeps only the K leading principal portfolios and zeroes out
            the rest. For the PP strategy these are the top K singular values
            of Pi, for the PEP strategy the K eigenvalues of Pi_s with the
            largest absolute values, and for the PAP strategy the top K pairs
            of Pi_a (rank 2K), capped at N // 2 pairs. If None (default),
            there is no rank restriction.

        p_norm: int, float
            Regularization parameter p in [1, inf] of Proposition 11: the
            exponent of the Schatten p-norm constraint ||L||_p <= 1. With
            1/p + 1/q = 1, the retained principal portfolios are weighted
            proportionally to lambda_k^(q-1). p = inf (default) is the
            operator-norm constraint of the unregularized solutions
            (Propositions 3, 6 and 8), which weights every retained portfolio
            equally. p = 2 weights them by their expected returns
            (L proportional to Pi' with no rank restriction), and p = 1 puts
            all the weight on the leading portfolio.

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

        # Regularization parameters of Proposition 11
        if rank is not None:
            assert isinstance(rank, (int, np.integer)) and 1 <= rank <= self.N, \
                f"`rank` must be None or an integer between 1 and N={self.N}"
        assert p_norm >= 1, "`p_norm` must be in the interval [1, inf]"
        self.rank = rank
        self.p_norm = p_norm
        self.q_norm = self._dual_exponent(p_norm)

        self.Pi = self._estimate_predictability_matrix(pi_weight)
        self.Pi_s, self.Pi_a, U, sv, V, lambdas_s, W, lambdas_a, X, Y = self._decompositions(self.Pi)

        pp_names = [f"PP {k + 1}" for k in range(self.N)]
        pep_names = [f"PEP {k + 1}" for k in range(self.N)]
        pap_names = [f"PAP {k + 1}" for k in range(self.N // 2)]

        # The singular-values / eigenvalues of each strategy are the expected returns
        self.singular_values = pd.Series(sv, index=pp_names, name="Singular Values")
        self.pep_eigenvalues = pd.Series(lambdas_s, index=pep_names, name="PEP Eigenvalues")
        self.pap_eigenvalues = pd.Series(lambdas_a, index=pap_names, name="PAP Eigenvalues")

        # Portfolio weights
        self.L, self.L_s, self.L_a, self.w, self.w_s, self.w_a = self._portfolio_weights(V, U, sv, lambdas_s, W, X, Y, lambdas_a)

    def plot_lambdas(self, size=5, title=None, save_path=None, color1="tab:blue", color2="tab:orange"):
        """
        Plots the singular values of the prediction matrix, the eigenvalues
        of its symmetric part, and the pair magnitudes of its antisymmetric
        part.

        Parameters
        ----------
        size: float
            Relative size of the chart

        title: str, optional
            Title for the chart

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)

        color1: str
            Color of the bars

        color2: str
            Color used to highlight the negative PEP eigenvalues
        """
        plt.figure(figsize=(size * (16 / 7.3), size))
        if title is not None:
            plt.suptitle(title)

        ax = plt.subplot2grid((1, 3), (0, 0))
        ax.set_title(r"Singular Values of $\Pi$" + "\n" + r"$E[PP_k]=\bar{\lambda_k}$")
        ax.bar(range(1, self.N + 1), self.singular_values.values, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("$k$-th singular value")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        ax = plt.subplot2grid((1, 3), (0, 1))
        ax.set_title(r"Eigenvalues of $\Pi_{s}$" + "\n" + r"$E[PEP_k]=\lambda_k^s$")
        colors = [color1 if v >= 0 else color2 for v in self.pep_eigenvalues.values]
        ax.bar(range(1, self.N + 1), self.pep_eigenvalues.values, color=colors)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("$k$-th eigenvalue")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        ax = plt.subplot2grid((1, 3), (0, 2))
        ax.set_title(r"Pair Magnitudes $\lambda^{a}_{k}$ of $\Pi_{a}$" + "\n" + r"$E[PAP_k] = 2\lambda_k^a$")
        ax.bar(range(1, self.N // 2 + 1), self.pap_eigenvalues.values, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("$k$-th paired magnitude")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_prediction_matrix(self, which="full", size=5, title=None, save_path=None, cmap="RdBu_r"):
        """
        Heatmap of the prediction matrix (or one of its symmetry
        components). Rows are the return leg and columns are the signal
        leg, so entry (i, j) measures how the signal of asset j predicts
        the return of asset i.

        Parameters
        ----------
        which: str
            "full" for Pi, "symmetric" for Pi_sym, "antisymmetric" for
            Pi_asym

        size: float
            Relative size of the chart

        title: str, optional
            Title for the chart

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)

        cmap: str
            Diverging colormap used by the heatmap
        """
        options = {
            "full": (self.Pi, r"$\Pi$"),
            "symmetric": (self.Pi_s, r"$\Pi_{s}$"),
            "antisymmetric": (self.Pi_a, r"$\Pi_{a}$"),
        }
        if which not in options:
            raise ValueError(
                f"which must be 'full', 'symmetric' or 'antisymmetric'; got {which!r}"
            )
        mat, default_title = options[which]

        plt.figure(figsize=(size * 1.2, size))
        vmax = np.abs(mat.values).max()
        ax = sns.heatmap(
            mat,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title if title is not None else default_title)
        ax.set_xlabel("Signal")
        ax.set_ylabel("Return")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

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

    @staticmethod
    def _dual_exponent(p):
        """
        Hölder conjugate q of the Schatten norm exponent p, 1/p + 1/q = 1,
        including the limiting cases p = 1 (q = inf) and p = inf (q = 1).
        """
        if p == 1:
            return np.inf
        elif np.isinf(p):
            return 1.0
        else:
            return p / (p - 1)

    def _regularized_weights(self, magnitudes, block_size=1):
        """
        Weights of the retained principal port folios in Proposition 11,
        """
        m_max = magnitudes.max()
        rel = magnitudes / m_max if m_max > 0 else np.ones_like(magnitudes)
        d = rel ** (self.q_norm - 1)  # note that 0^0 = 1 when q = 1 (p = inf)

        if np.isinf(self.p_norm):
            norm = d.max()
        else:
            norm = (block_size * np.sum(d ** self.p_norm)) ** (1 / self.p_norm)

        return d / norm

    def _portfolio_weights(self, V, U, sv, lambdas_s, W, X, Y, lambdas_a):
        # With the default p = inf and no rank restriction, the solutions
        # below reduce to the unregularized optimal strategies of
        # Propositions 3, 6 and 8, since every retained weight equals one.
        K = self.N if self.rank is None else self.rank
        K_a = min(K, self.N // 2)  # PAP components come in pairs

        # Optimal linear strategy (Proposition 11 (i)):
        # L = c sum_{k<=K} sv_k^(q-1) v_k u_k'. Singular values from svd are
        # already sorted in descending order.
        d = self._regularized_weights(sv[:K])
        L = V[:, :K] @ np.diag(d) @ U[:, :K].T

        # Optimal symmetric strategy (Proposition 11 (ii)):
        # L_s = c sum_{k in K} |lambda_k^s|^(q-1) sign(lambda_k^s) w_k w_k',
        # where the retained set holds the K largest ABSOLUTE eigenvalues,
        # which can include negative ones.
        idx = np.argsort(-np.abs(lambdas_s), kind="stable")[:K]
        d_s = self._regularized_weights(np.abs(lambdas_s[idx])) * np.sign(lambdas_s[idx])
        L_s = W[:, idx] @ np.diag(d_s) @ W[:, idx].T

        # Optimal antisymmetric strategy (Proposition 11 (iii)):
        # L_a = c sum_{k<=K} (lambda_k^a)^(q-1) (x_k y_k' - y_k x_k')
        # The pair magnitudes are already sorted in descending order.
        d_a = self._regularized_weights(lambdas_a[:K_a], block_size=2)
        L_a = (X[:, :K_a] @ np.diag(d_a) @ Y[:, :K_a].T
               - Y[:, :K_a] @ np.diag(d_a) @ X[:, :K_a].T)

        # Weights
        w = self.signals @ pd.DataFrame(L, index=self.assets, columns=self.assets)
        w_s = self.signals @ pd.DataFrame(L_s, index=self.assets, columns=self.assets)
        w_a = self.signals @ pd.DataFrame(L_a, index=self.assets, columns=self.assets)

        return L, L_s, L_a, w, w_s, w_a
