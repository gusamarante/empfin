import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from numpy.linalg import eigh, svd
from scipy.linalg import schur
from tqdm import tqdm


def momentum_signal(returns, lookback=12, skip=0):
    """
    Builds a simple momentum signal from a panel of returns. The signal at
    date t for each asset is the sum of its returns from t - skip - lookback + 1
    to t - skip. The dates with insufficient history are left as NaN.

    Parameters
    ----------
    returns: pandas.DataFrame
        Timeseries of asset returns

    lookback: int
        Number of periods summed to build the momentum signal

    skip: int
        Number of most recent periods to skip. `skip=1` with monthly data
        gives the classical momentum convention that avoids the 1-month
        reversal effect

    Returns
    -------
    pandas.DataFrame
        Timeseries of the momentum signal, same shape as `returns`
    """
    return returns.rolling(lookback).sum().shift(skip)


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
            f"signal_transform must be 'rank', 'zscore' or 'none'; got {signal_transform!r}"
        )


def _build_pairs(returns, signals, signal_transform):
    """
    Aligns the panels, applies the signal transformation, and pairs the
    signal known at date t with the return realized at t + 1. Both returned
    DataFrames are indexed by the return date t + 1, and rows with any
    missing value in either panel are dropped.
    """
    assert isinstance(returns, pd.DataFrame), "`returns` must be a pandas.DataFrame"
    assert isinstance(signals, pd.DataFrame), "`signals` must be a pandas.DataFrame"
    assert set(returns.columns) == set(signals.columns), \
        "`returns` and `signals` must have the same columns"

    signals = signals[returns.columns]
    common_index = returns.index.intersection(signals.index)
    returns = returns.loc[common_index]
    signals = signals.loc[common_index]

    S = _transform_signals(signals, signal_transform)

    R_fwd = returns.iloc[1:]
    S_lag = S.iloc[:-1].set_axis(R_fwd.index)

    valid = (~R_fwd.isna().any(axis=1)) & (~S_lag.isna().any(axis=1))
    return R_fwd.loc[valid], S_lag.loc[valid]


def _antisymmetric_pairs(Pi_a):
    """
    Real canonical decomposition of the antisymmetric matrix

        Pi_a = sum_k lambda_k * (x_k @ y_k' - y_k @ x_k')

    with lambda_k > 0 and {x_k, y_k} orthonormal, computed from the real
    Schur form. The pairs are sorted by decreasing lambda_k and padded with
    NaN when Pi_a is rank deficient, so exactly floor(N / 2) pairs are
    always returned.

    Returns
    -------
    lambdas: numpy.ndarray
        Vector with the floor(N / 2) values of lambda_k

    X: numpy.ndarray
        Matrix whose k-th column is x_k

    Y: numpy.ndarray
        Matrix whose k-th column is y_k
    """
    N = Pi_a.shape[0]
    n_pairs = N // 2
    T_mat, Q = schur(Pi_a, output="real")
    tol = 1e-12 * max(np.abs(T_mat).max(), 1.0)

    lambdas, xs, ys = [], [], []
    i = 0
    while i < N - 1:
        b = T_mat[i, i + 1]
        if abs(T_mat[i + 1, i]) > tol or abs(b) > tol:  # 2x2 block found
            if b >= 0:
                lambdas.append(b)
                xs.append(Q[:, i])
                ys.append(Q[:, i + 1])
            else:
                lambdas.append(-b)
                xs.append(Q[:, i + 1])
                ys.append(Q[:, i])
            i += 2
        else:  # 1x1 zero block
            i += 1

    order = np.argsort(lambdas)[::-1]
    lambdas = [lambdas[j] for j in order]
    xs = [xs[j] for j in order]
    ys = [ys[j] for j in order]

    while len(lambdas) < n_pairs:  # Pad rank-deficient cases
        lambdas.append(np.nan)
        xs.append(np.full(N, np.nan))
        ys.append(np.full(N, np.nan))

    lambdas = np.array(lambdas[:n_pairs])
    X = np.column_stack(xs[:n_pairs])
    Y = np.column_stack(ys[:n_pairs])
    return lambdas, X, Y


def _decompose(Pi):
    """
    Computes the three decompositions of the prediction matrix used by the
    principal portfolios framework: the SVD of Pi itself (principal
    portfolios), the eigendecomposition of the symmetric part (principal
    exposure portfolios), and the real canonical form of the antisymmetric
    part (principal alpha portfolios). Returns a dict of numpy arrays.
    """
    Pi_s = (Pi + Pi.T) / 2
    Pi_a = (Pi - Pi.T) / 2

    U, sv, Vt = svd(Pi)
    V = Vt.T

    eigval_s, W = eigh(Pi_s)
    order = np.argsort(eigval_s)[::-1]
    eigval_s = eigval_s[order]
    W = W[:, order]

    lambdas_a, X, Y = _antisymmetric_pairs(Pi_a)

    return {
        "Pi_s": Pi_s,
        "Pi_a": Pi_a,
        "U": U,
        "sv": sv,
        "V": V,
        "eigval_s": eigval_s,
        "W": W,
        "lambdas_a": lambdas_a,
        "X": X,
        "Y": Y,
    }


def _component_returns(R, S, dec):
    """
    One-period returns of every principal portfolio given paired arrays of
    returns `R` and signals `S` (both n_obs x N) and a decomposition `dec`
    from `_decompose`. Returns the tuple (pp, pep, pap, simple) of numpy
    arrays.
    """
    pp = (S @ dec["V"]) * (R @ dec["U"])
    pep = (S @ dec["W"]) * (R @ dec["W"])
    pap = (S @ dec["Y"]) * (R @ dec["X"]) - (S @ dec["X"]) * (R @ dec["Y"])
    simple = (S * R).sum(axis=1)
    return pp, pep, pap, simple


class PrincipalPortfolios:
    """
    References:
        Kelly, Bryan T., Semyon Malamud, and Lasse Heje Pedersen (2023)
        "Principal Portfolios"
        The Journal of Finance, 78(1), 347-387
        https://doi.org/10.1111/jofi.13199
    """

    def __init__(self, returns, signals, signal_transform="rank"):
        """
        Full-sample (in-sample) estimation of the principal portfolios of
        Kelly, Malamud and Pedersen (2023).

        The framework uses the signals of all assets to predict the return
        of each individual asset, including cross-predictability. All the
        information relevant for linear strategies is summarized by the
        prediction matrix

            Pi = E[R_{t+1} @ S_t']

        whose (i, j) entry measures how much the signal of asset j predicts
        the return of asset i. A linear strategy holds the positions
        w_t = L @ S_t and earns E[w_t' R_{t+1}] = tr(L' Pi). The optimal
        strategy subject to a unit bound on the largest singular value of L
        is L = sum_k u_k @ v_k', built from the SVD

            Pi = sum_k sv_k * u_k @ v_k'

        Its return decomposes into the "principal portfolios"

            PP_k = (S_t' v_k) * (u_k' R_{t+1})

        which are portfolios with positions u_k, scaled over time by the
        timing signal S_t' v_k. In sample, E[PP_k] equals the k-th singular
        value.

        The symmetry decomposition Pi = Pi_s + Pi_a, with the symmetric part
        Pi_s = (Pi + Pi') / 2 and the antisymmetric part Pi_a = (Pi - Pi') / 2,
        splits the problem into beta and alpha:

        - The eigenvectors w_k of Pi_s give the "principal exposure
          portfolios" PEP_k = (S_t' w_k) * (w_k' R_{t+1}), with in-sample
          expected return equal to the eigenvalue. PEPs are factor-exposure
          strategies. Under the null of no alpha (signals only predict
          returns through factor exposure), Pi_s is positive semidefinite,
          so negative eigenvalues are evidence of mispricing.

        - The antisymmetric part admits the real canonical form
          Pi_a = sum_k lambda_k * (x_k @ y_k' - y_k @ x_k') with
          lambda_k > 0, giving the "principal alpha portfolios"

              PAP_k = (S_t' y_k) * (x_k' R_{t+1}) - (S_t' x_k) * (y_k' R_{t+1})

          with in-sample expected return 2 * lambda_k. PAPs are pure
          cross-predictability strategies (buy asset x on the signal of
          asset y and short asset y on the signal of asset x) and have zero
          factor exposure under the model's null. A nonzero Pi_a is direct
          evidence of alpha.

        The simple own-signal factor S_t' R_{t+1} (each asset held in
        proportion to its own signal) equals the sum of all PEP returns,
        since the eigenvectors of Pi_s form an orthonormal basis.

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

        Attributes
        ----------
        T: int
            Number of (signal, next-period return) pairs used

        N: int
            Number of assets

        R: pandas.DataFrame
            Aligned panel of returns, indexed by the return date t + 1

        S: pandas.DataFrame
            Aligned panel of transformed signals. Row t + 1 holds the signal
            known at date t, paired with the return of date t + 1

        Pi: pandas.DataFrame
            Estimated prediction matrix (1 / T) * sum_t R_{t+1} @ S_t'.
            Rows are the return leg, columns are the signal leg

        Pi_sym: pandas.DataFrame
            Symmetric part of the prediction matrix

        Pi_asym: pandas.DataFrame
            Antisymmetric part of the prediction matrix

        singular_values: pandas.Series
            Singular values of `Pi`, which equal the in-sample expected
            returns of the principal portfolios

        pep_eigenvalues: pandas.Series
            Eigenvalues of `Pi_sym` in decreasing order, which equal the
            in-sample expected returns of the PEPs

        pap_eigenvalues: pandas.Series
            The floor(N / 2) values of lambda_k from the canonical form of
            `Pi_asym` in decreasing order. The in-sample expected return of
            PAP_k is 2 * lambda_k

        pp_positions: pandas.DataFrame
            Position portfolios u_k of each principal portfolio (assets in
            rows, components in columns)

        pp_timing: pandas.DataFrame
            Timing portfolios v_k of each principal portfolio. The position
            in u_k at time t is scaled by S_t' v_k

        pep_weights: pandas.DataFrame
            Eigenvector portfolios w_k of the PEPs

        pap_x, pap_y: pandas.DataFrame
            The orthonormal pairs (x_k, y_k) defining each PAP

        pp_returns: pandas.DataFrame
            In-sample timeseries of the PP returns

        pep_returns: pandas.DataFrame
            In-sample timeseries of the PEP returns

        pap_returns: pandas.DataFrame
            In-sample timeseries of the PAP returns

        simple_returns: pandas.Series
            In-sample timeseries of the simple own-signal factor S_t' R_{t+1}
        """
        self.signal_transform = signal_transform
        self.R, self.S = _build_pairs(returns, signals, signal_transform)

        self.T, self.N = self.R.shape
        assert self.T >= 2, "Not enough valid (signal, return) pairs"

        assets = self.R.columns
        R_vals = self.R.values
        S_vals = self.S.values

        Pi = (R_vals.T @ S_vals) / self.T
        self.Pi = pd.DataFrame(Pi, index=assets, columns=assets)

        dec = _decompose(Pi)
        self._dec = dec
        self.Pi_sym = pd.DataFrame(dec["Pi_s"], index=assets, columns=assets)
        self.Pi_asym = pd.DataFrame(dec["Pi_a"], index=assets, columns=assets)

        pp_names = [f"PP {k + 1}" for k in range(self.N)]
        pep_names = [f"PEP {k + 1}" for k in range(self.N)]
        pap_names = [f"PAP {k + 1}" for k in range(self.N // 2)]

        self.singular_values = pd.Series(dec["sv"], index=pp_names, name="Singular Values")
        self.pep_eigenvalues = pd.Series(dec["eigval_s"], index=pep_names, name="Eigenvalues")
        self.pap_eigenvalues = pd.Series(dec["lambdas_a"], index=pap_names, name="Lambdas")

        self.pp_positions = pd.DataFrame(dec["U"], index=assets, columns=pp_names)
        self.pp_timing = pd.DataFrame(dec["V"], index=assets, columns=pp_names)
        self.pep_weights = pd.DataFrame(dec["W"], index=assets, columns=pep_names)
        self.pap_x = pd.DataFrame(dec["X"], index=assets, columns=pap_names)
        self.pap_y = pd.DataFrame(dec["Y"], index=assets, columns=pap_names)

        pp, pep, pap, simple = _component_returns(R_vals, S_vals, dec)
        self.pp_returns = pd.DataFrame(pp, index=self.R.index, columns=pp_names)
        self.pep_returns = pd.DataFrame(pep, index=self.R.index, columns=pep_names)
        self.pap_returns = pd.DataFrame(pap, index=self.R.index, columns=pap_names)
        self.simple_returns = pd.Series(simple, index=self.R.index, name="Simple Factor")

    def optimal_strategy(self, kind="pp", n_components=None):
        """
        In-sample returns of the optimal linear strategy built from the
        first `n_components` components of the chosen decomposition.

        Parameters
        ----------
        kind: str
            "pp" for the overall optimal strategy L = sum_k u_k @ v_k'
            (sum of the principal portfolios), "pep" for the optimal
            symmetric strategy L = sum_k sign(eig_k) * w_k @ w_k' (goes long
            the PEPs with positive eigenvalues and shorts the ones with
            negative eigenvalues), or "pap" for the optimal antisymmetric
            strategy (sum of the PAPs)

        n_components: int, optional
            Number of components used. If None, uses all of them. For
            "pep", the components are picked in order of absolute
            eigenvalue

        Returns
        -------
        pandas.Series
            Timeseries of the strategy returns
        """
        if kind == "pp":
            rets = self.pp_returns
            if n_components is None:
                n_components = rets.shape[1]
            out = rets.iloc[:, :n_components].sum(axis=1)
        elif kind == "pep":
            rets = self.pep_returns * np.sign(self.pep_eigenvalues)
            if n_components is None:
                n_components = rets.shape[1]
            order = self.pep_eigenvalues.abs().sort_values(ascending=False).index
            out = rets[order[:n_components]].sum(axis=1)
        elif kind == "pap":
            rets = self.pap_returns
            if n_components is None:
                n_components = rets.shape[1]
            out = rets.iloc[:, :n_components].sum(axis=1)
        else:
            raise ValueError(f"kind must be 'pp', 'pep' or 'pap'; got {kind!r}")

        return out.rename(f"Optimal {kind.upper()} (K={n_components})")

    def linear_rule(self, kind="pp", n_components=None):
        """
        The matrix L of the optimal linear trading rule w_t = L @ S_t built
        from the first `n_components` components of the chosen
        decomposition. Same conventions as `optimal_strategy`.

        Returns
        -------
        pandas.DataFrame
            The N x N matrix L. Rows are positions, columns are signals
        """
        if kind == "pp":
            if n_components is None:
                n_components = self.N
            U = self.pp_positions.values[:, :n_components]
            V = self.pp_timing.values[:, :n_components]
            L = U @ V.T
        elif kind == "pep":
            if n_components is None:
                n_components = self.N
            order = self.pep_eigenvalues.abs().sort_values(ascending=False).index
            cols = [self.pep_eigenvalues.index.get_loc(c) for c in order[:n_components]]
            W = self.pep_weights.values[:, cols]
            signs = np.sign(self.pep_eigenvalues.values[cols])
            L = (W * signs) @ W.T
        elif kind == "pap":
            if n_components is None:
                n_components = self.N // 2
            X = self.pap_x.values[:, :n_components]
            Y = self.pap_y.values[:, :n_components]
            L = X @ Y.T - Y @ X.T
        else:
            raise ValueError(f"kind must be 'pp', 'pep' or 'pap'; got {kind!r}")

        return pd.DataFrame(L, index=self.Pi.index, columns=self.Pi.columns)

    def factor_regressions(self, n_components=3, factors=None):
        """
        Regresses the returns of the leading principal portfolios on the
        simple own-signal factor (and optional extra factors) to measure
        their alphas and factor exposures. Under the model's null, the PAPs
        have zero exposure to the symmetric (factor) component, so this
        regression makes the alpha/beta split visible.

        Parameters
        ----------
        n_components: int
            Number of leading components of each type to include. The last
            `n_components` PEPs (the most negative eigenvalues) are also
            included, with their returns flipped ("Short PEP N")

        factors: pandas.DataFrame, optional
            Additional factors to include in the regressions, e.g. the
            Fama-French factors. Must be indexed by the same dates as the
            returns

        Returns
        -------
        pandas.DataFrame
            One row per strategy with the per-period alpha, its t-stat, the
            betas on each factor, and the regression R2
        """
        strategies = []
        for k in range(min(n_components, self.N)):
            strategies.append(self.pp_returns.iloc[:, k])
            strategies.append(self.pep_returns.iloc[:, k])
        for k in range(min(n_components, self.N)):
            flipped = -self.pep_returns.iloc[:, self.N - 1 - k]
            strategies.append(flipped.rename(f"Short PEP {self.N - k}"))
        for k in range(min(n_components, self.N // 2)):
            strategies.append(self.pap_returns.iloc[:, k])

        X = self.simple_returns.to_frame()
        if factors is not None:
            X = pd.concat([X, factors], axis=1)

        results = []
        for strat in strategies:
            data = pd.concat([strat, X], axis=1).dropna()
            model = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:]))
            res = model.fit()
            row = {
                "Mean": strat.mean(),
                "Alpha": res.params["const"],
                "t(Alpha)": res.tvalues["const"],
            }
            for fac in X.columns:
                row[f"Beta {fac}"] = res.params[fac]
            row["R2"] = res.rsquared
            results.append(pd.Series(row, name=strat.name))

        return pd.DataFrame(results)

    def plot_spectrum(
            self,
            size=5,
            title=None,
            save_path=None,
            color1="tab:blue",
            color2="tab:orange",
    ):
        """
        Plots the singular values of the prediction matrix, the eigenvalues
        of its symmetric part, and the pair magnitudes of its antisymmetric
        part. Each of these equals (or is half of, for the PAPs) the
        in-sample expected return of the corresponding principal portfolio.
        Negative eigenvalues of the symmetric part and nonzero values for
        the antisymmetric part are evidence against the no-alpha null.

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
        ax.set_title(r"Singular Values of $\Pi$" + "\n" + r"$E[PP_k]$")
        ax.bar(range(1, self.N + 1), self.singular_values.values, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("$k$")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        ax = plt.subplot2grid((1, 3), (0, 1))
        ax.set_title(r"Eigenvalues of $\Pi_{s}$" + "\n" + r"$E[PEP_k]$")
        colors = [color1 if v >= 0 else color2 for v in self.pep_eigenvalues.values]
        ax.bar(range(1, self.N + 1), self.pep_eigenvalues.values, color=colors)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("$k$")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        ax = plt.subplot2grid((1, 3), (0, 2))
        ax.set_title(r"Pair Magnitudes $\lambda_k$ of $\Pi_{a}$" + "\n" + r"$E[PAP_k] = 2\lambda_k$")
        ax.bar(range(1, self.N // 2 + 1), self.pap_eigenvalues.values, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("$k$")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_prediction_matrix(
            self,
            which="full",
            size=6,
            title=None,
            save_path=None,
            cmap="RdBu_r",
    ):
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
            "symmetric": (self.Pi_sym, r"$\Pi_{s}$"),
            "antisymmetric": (self.Pi_asym, r"$\Pi_{a}$"),
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

    def plot_cumulative_returns(
            self,
            strategies=None,
            unit_vol=False,
            size=6,
            title=None,
            save_path=None,
    ):
        """
        Plots the cumulative (summed) returns of the chosen strategies
        together with the simple own-signal factor.

        Parameters
        ----------
        strategies: list of str, optional
            Names of the components to plot, from the columns of
            `pp_returns`, `pep_returns` and `pap_returns` (e.g. "PP 1",
            "PEP 25", "PAP 2"). "Simple Factor" is also accepted. If None,
            plots the first PP, the first and last PEPs, the first PAP, and
            the simple factor

        unit_vol: bool
            If True, each strategy is scaled to unit full-sample volatility
            before cumulating, making the paths comparable

        size: float
            Relative size of the chart

        title: str, optional
            Title for the chart

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)
        """
        if strategies is None:
            strategies = ["PP 1", "PEP 1", f"PEP {self.N}", "PAP 1", "Simple Factor"]

        plt.figure(figsize=(size * (16 / 7.3), size))
        ax = plt.subplot2grid((1, 1), (0, 0))

        for name in strategies:
            series = self._get_strategy(name)
            if unit_vol:
                series = series / series.std()
            ax.plot(series.index, series.cumsum().values, lw=1.2, label=name)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title if title is not None else "Cumulative Returns")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="best")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def _get_strategy(self, name):
        if name == "Simple Factor":
            return self.simple_returns
        for rets in (self.pp_returns, self.pep_returns, self.pap_returns):
            if name in rets.columns:
                return rets[name]
        raise KeyError(f"Unknown strategy {name!r}")


class PrincipalPortfoliosBacktest:
    """
    References:
        Kelly, Bryan T., Semyon Malamud, and Lasse Heje Pedersen (2023)
        "Principal Portfolios"
        The Journal of Finance, 78(1), 347-387
        https://doi.org/10.1111/jofi.13199
    """

    def __init__(self, returns, signals, window=120, signal_transform="rank"):
        """
        Out-of-sample rolling backtest of the principal portfolios of
        Kelly, Malamud and Pedersen (2023). See `PrincipalPortfolios` for
        the description of the framework.

        At each date t, the prediction matrix is estimated using only the
        trailing `window` pairs of signals and next-period returns, its
        decompositions are computed, and the resulting portfolio weights
        are combined with the signal observed at t and the return realized
        at t + 1. The component returns are therefore fully out of sample.
        The paper uses a 120-month trailing window in its monthly
        applications.

        Note that the ordering of the components ("PP 1", "PEP 2", ...) is
        re-computed at every estimation date, so a column tracks the k-th
        largest component at each date, not a fixed portfolio.

        Parameters
        ----------
        returns: pandas.DataFrame
            Timeseries of asset (excess) returns

        signals: pandas.DataFrame
            Timeseries of the predictive signals, with the same conventions
            as in `PrincipalPortfolios`

        window: int
            Number of trailing (signal, return) pairs used to estimate the
            prediction matrix at each date

        signal_transform: str
            Cross-sectional transformation applied to the signals ("rank",
            "zscore" or "none"), as in `PrincipalPortfolios`

        Attributes
        ----------
        T: int
            Total number of (signal, return) pairs in the sample

        T_oos: int
            Number of out-of-sample periods, T - window

        N: int
            Number of assets

        window: int
            The trailing estimation window used

        pp_returns: pandas.DataFrame
            Out-of-sample timeseries of the returns of each principal
            portfolio

        pep_returns: pandas.DataFrame
            Out-of-sample timeseries of the returns of each principal
            exposure portfolio

        pap_returns: pandas.DataFrame
            Out-of-sample timeseries of the returns of each principal alpha
            portfolio

        simple_returns: pandas.Series
            Out-of-sample timeseries of the simple own-signal factor

        singular_values_t: pandas.DataFrame
            Timeseries of the singular values of the rolling prediction
            matrix, indexed by the date of the out-of-sample return they
            were used for

        pep_eigenvalues_t: pandas.DataFrame
            Timeseries of the eigenvalues of the symmetric part

        pap_eigenvalues_t: pandas.DataFrame
            Timeseries of the pair magnitudes of the antisymmetric part
        """
        assert isinstance(window, (int, np.integer)) and window > 1, \
            "`window` must be an integer greater than 1"

        self.window = window
        self.signal_transform = signal_transform
        self.R, self.S = _build_pairs(returns, signals, signal_transform)

        self.T, self.N = self.R.shape
        assert self.T > window, \
            f"Not enough observations (T={self.T}) for window={window}"
        self.T_oos = self.T - window

        R_vals = self.R.values
        S_vals = self.S.values

        pp_names = [f"PP {k + 1}" for k in range(self.N)]
        pep_names = [f"PEP {k + 1}" for k in range(self.N)]
        pap_names = [f"PAP {k + 1}" for k in range(self.N // 2)]

        pp_rets = np.empty((self.T_oos, self.N))
        pep_rets = np.empty((self.T_oos, self.N))
        pap_rets = np.empty((self.T_oos, self.N // 2))
        simple_rets = np.empty(self.T_oos)
        sv_t = np.empty((self.T_oos, self.N))
        eig_s_t = np.empty((self.T_oos, self.N))
        lam_a_t = np.empty((self.T_oos, self.N // 2))

        for i, t in enumerate(tqdm(range(window, self.T), desc="Rolling estimation")):
            Pi = (R_vals[t - window:t].T @ S_vals[t - window:t]) / window
            dec = _decompose(Pi)
            pp, pep, pap, simple = _component_returns(
                R_vals[[t]], S_vals[[t]], dec,
            )
            pp_rets[i] = pp[0]
            pep_rets[i] = pep[0]
            pap_rets[i] = pap[0]
            simple_rets[i] = simple[0]
            sv_t[i] = dec["sv"]
            eig_s_t[i] = dec["eigval_s"]
            lam_a_t[i] = dec["lambdas_a"]

        oos_index = self.R.index[window:]
        self.pp_returns = pd.DataFrame(pp_rets, index=oos_index, columns=pp_names)
        self.pep_returns = pd.DataFrame(pep_rets, index=oos_index, columns=pep_names)
        self.pap_returns = pd.DataFrame(pap_rets, index=oos_index, columns=pap_names)
        self.simple_returns = pd.Series(simple_rets, index=oos_index, name="Simple Factor")
        self.singular_values_t = pd.DataFrame(sv_t, index=oos_index, columns=pp_names)
        self.pep_eigenvalues_t = pd.DataFrame(eig_s_t, index=oos_index, columns=pep_names)
        self.pap_eigenvalues_t = pd.DataFrame(lam_a_t, index=oos_index, columns=pap_names)

    def optimal_strategy(self, kind="pp", n_components=None):
        """
        Out-of-sample returns of the optimal linear strategy built from the
        first `n_components` components of the chosen decomposition,
        re-estimated at each date.

        Parameters
        ----------
        kind: str
            "pp" for the overall optimal strategy (sum of the principal
            portfolios), "pep" for the optimal symmetric strategy (each
            PEP held with the sign of its eigenvalue at that date, picked
            in order of absolute eigenvalue), or "pap" for the optimal
            antisymmetric strategy (sum of the PAPs)

        n_components: int, optional
            Number of components used. If None, uses all of them

        Returns
        -------
        pandas.Series
            Timeseries of the out-of-sample strategy returns
        """
        if kind == "pp":
            if n_components is None:
                n_components = self.N
            out = self.pp_returns.iloc[:, :n_components].sum(axis=1)
        elif kind == "pep":
            if n_components is None:
                n_components = self.N
            eig = self.pep_eigenvalues_t.values
            signed = self.pep_returns.values * np.sign(eig)
            order = np.argsort(-np.abs(eig), axis=1)[:, :n_components]
            out = pd.Series(
                data=np.take_along_axis(signed, order, axis=1).sum(axis=1),
                index=self.pep_returns.index,
            )
        elif kind == "pap":
            if n_components is None:
                n_components = self.N // 2
            out = self.pap_returns.iloc[:, :n_components].sum(axis=1)
        else:
            raise ValueError(f"kind must be 'pp', 'pep' or 'pap'; got {kind!r}")

        return out.rename(f"Optimal {kind.upper()} (K={n_components})")

    def performance_summary(self, n_components=3, factors=None, annualization=12):
        """
        Summary of the out-of-sample performance of the leading principal
        portfolios of each type, the last (most negative eigenvalue) PEPs
        held short, and the simple own-signal factor. Alphas and betas come
        from a regression of each strategy on the simple factor and any
        additional `factors`.

        Parameters
        ----------
        n_components: int
            Number of leading components of each type to include

        factors: pandas.DataFrame, optional
            Additional factors to include in the alpha regressions. Must be
            indexed by the same dates as the returns

        annualization: int
            Number of periods per year used to annualize the mean, the
            volatility and the Sharpe ratio (12 for monthly data)

        Returns
        -------
        pandas.DataFrame
            One row per strategy with the annualized mean and volatility,
            the annualized Sharpe ratio, the per-period alpha and its
            t-stat, the betas, and the regression R2
        """
        strategies = [self.simple_returns]
        for k in range(min(n_components, self.N)):
            strategies.append(self.pp_returns.iloc[:, k])
            strategies.append(self.pep_returns.iloc[:, k])
        for k in range(min(n_components, self.N)):
            flipped = -self.pep_returns.iloc[:, self.N - 1 - k]
            strategies.append(flipped.rename(f"Short PEP {self.N - k}"))
        for k in range(min(n_components, self.N // 2)):
            strategies.append(self.pap_returns.iloc[:, k])

        X = self.simple_returns.to_frame()
        if factors is not None:
            X = pd.concat([X, factors], axis=1)

        results = []
        for strat in strategies:
            row = {
                "Mean": strat.mean() * annualization,
                "Vol": strat.std() * np.sqrt(annualization),
                "Sharpe": (strat.mean() / strat.std()) * np.sqrt(annualization),
            }
            if strat.name == self.simple_returns.name:
                row["Alpha"] = np.nan
                row["t(Alpha)"] = np.nan
                for fac in X.columns:
                    row[f"Beta {fac}"] = np.nan
                row["R2"] = np.nan
            else:
                data = pd.concat([strat, X], axis=1).dropna()
                model = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:]))
                res = model.fit()
                row["Alpha"] = res.params["const"]
                row["t(Alpha)"] = res.tvalues["const"]
                for fac in X.columns:
                    row[f"Beta {fac}"] = res.params[fac]
                row["R2"] = res.rsquared
            results.append(pd.Series(row, name=strat.name))

        return pd.DataFrame(results)

    def plot_cumulative_returns(
            self,
            strategies=None,
            unit_vol=False,
            size=6,
            title=None,
            save_path=None,
    ):
        """
        Plots the cumulative (summed) out-of-sample returns of the chosen
        strategies together with the simple own-signal factor.

        Parameters
        ----------
        strategies: list of str, optional
            Names of the components to plot, from the columns of
            `pp_returns`, `pep_returns` and `pap_returns` (e.g. "PP 1",
            "PEP 25", "PAP 2"). "Simple Factor" is also accepted. If None,
            plots the first PP, the first and last PEPs, the first PAP, and
            the simple factor

        unit_vol: bool
            If True, each strategy is scaled to unit full-sample volatility
            before cumulating, making the paths comparable

        size: float
            Relative size of the chart

        title: str, optional
            Title for the chart

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)
        """
        if strategies is None:
            strategies = ["PP 1", "PEP 1", f"PEP {self.N}", "PAP 1", "Simple Factor"]

        plt.figure(figsize=(size * (16 / 7.3), size))
        ax = plt.subplot2grid((1, 1), (0, 0))

        for name in strategies:
            series = self._get_strategy(name)
            if unit_vol:
                series = series / series.std()
            ax.plot(series.index, series.cumsum().values, lw=1.2, label=name)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title if title is not None else "Cumulative Out-of-Sample Returns")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="best")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_eigenvalues_timeseries(
            self,
            n_components=3,
            size=5,
            title=None,
            save_path=None,
    ):
        """
        Plots the timeseries of the leading singular values, symmetric-part
        eigenvalues (largest and smallest), and antisymmetric-part pair
        magnitudes of the rolling prediction matrix. These are the
        conditional in-sample expected returns of the corresponding
        principal portfolios at each estimation date.

        Parameters
        ----------
        n_components: int
            Number of leading components of each type to plot

        size: float
            Relative size of the chart

        title: str, optional
            Title for the chart

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)
        """
        plt.figure(figsize=(size * (16 / 7.3), size))
        if title is not None:
            plt.suptitle(title)

        ax = plt.subplot2grid((1, 3), (0, 0))
        ax.set_title(r"Singular Values of $\Pi$")
        for k in range(min(n_components, self.N)):
            ax.plot(self.singular_values_t.iloc[:, k], lw=1.0, label=f"PP {k + 1}")
        ax.axhline(0, color="black", lw=0.5)
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="best")

        ax = plt.subplot2grid((1, 3), (0, 1))
        ax.set_title(r"Eigenvalues of $\Pi_{s}$")
        for k in range(min(n_components, self.N)):
            ax.plot(self.pep_eigenvalues_t.iloc[:, k], lw=1.0, label=f"PEP {k + 1}")
            ax.plot(
                self.pep_eigenvalues_t.iloc[:, self.N - 1 - k],
                lw=1.0,
                ls="--",
                label=f"PEP {self.N - k}",
            )
        ax.axhline(0, color="black", lw=0.5)
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="best")

        ax = plt.subplot2grid((1, 3), (0, 2))
        ax.set_title(r"Pair Magnitudes of $\Pi_{a}$")
        for k in range(min(n_components, self.N // 2)):
            ax.plot(self.pap_eigenvalues_t.iloc[:, k], lw=1.0, label=f"PAP {k + 1}")
        ax.axhline(0, color="black", lw=0.5)
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="best")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def _get_strategy(self, name):
        if name == "Simple Factor":
            return self.simple_returns
        for rets in (self.pp_returns, self.pep_returns, self.pap_returns):
            if name in rets.columns:
                return rets[name]
        raise KeyError(f"Unknown strategy {name!r}")
