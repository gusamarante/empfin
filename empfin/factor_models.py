import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.linalg import eigvals, inv, svd
from scipy.linalg import cholesky
from scipy.stats import (
    f,
    invgamma,
    invwishart,
    matrix_normal,
    multivariate_normal,
    norm,
)
from sklearn.decomposition import PCA
import statsmodels.api as sm
from tqdm import tqdm


class TimeseriesReg:

    def __init__(self, assets, factors):
        """
        Estimates for a linear factor model using time series regressions.

            r_i = alpha_i + beta_i * f + eps_i

        All factors f must be excess returns.

        Parameters
        ----------
        assets: pandas.DataFrame
            timeseries of test assets returns

        factors: pandas.DataFrame
            timeseries of the factor portfolios returns

        Notes
        -----
        Check section 12.1 of Cochrane (2009) for more details
        """

        assert assets.index.equals(factors.index), \
            "Indexes of `assets` and `factors` must be the same"

        self.T = assets.shape[0]
        self.N = assets.shape[1]
        self.K = factors.shape[1]

        # Risk premia estimates are just the historical means
        self.lambdas = factors.mean()
        self.Omega = factors.cov()

        # OLS
        factors = pd.concat([pd.Series(1, index=factors.index, name="alpha"), factors], axis=1)

        X = factors.values
        Bhat = inv(X.T @ X) @ X.T @ assets.values
        self.resids = pd.DataFrame(
            data=assets.values - X @ Bhat,
            index=assets.index,
            columns=assets.columns,
        )
        self.params = pd.DataFrame(
            data=Bhat,
            index=factors.columns,
            columns=assets.columns,
        )
        self.Sigma = self.resids.values.T @ self.resids.values

    def grs_test(self):
        """
        Runs the Gibbons-Ross-Shanken test to evaluate if all alphas are
        jointly equal to zero.

        Returns
        -------
        grs: float
            Gibbons-Ross-Shanken statistic

        pvalue: float
            p-value of the Gibbons-Ross-Shanken statistic
        """
        f1 = (self.T - self.N - self.K) / self.N
        f2 = 1 / (1 + self.lambdas.T @ inv(self.Omega) @ self.lambdas)
        alpha = self.params.loc['alpha']
        f3 = alpha.T @ inv(self.Sigma) @ alpha
        grs = f1 * f2 * f3
        pvalue = 1 - f.cdf(grs, dfn=self.N, dfd=self.T - self.N - self.K)
        return grs, pvalue

class TwoPassOLS:

    def __init__(self, assets, factors, cs_const=False):
        # TODO DOcumentation

        # First stage is the timeseries regression
        ts_reg = TimeseriesReg(assets.copy(), factors.copy())
        self.T = ts_reg.T
        self.N = ts_reg.N
        self.K = ts_reg.K
        self.betas = ts_reg.params.drop('alpha')
        self.avg_ret = assets.mean()
        self.Sigma = ts_reg.Sigma
        self.Omega = ts_reg.Omega

        if cs_const:
            self.betas = pd.concat([pd.DataFrame(data=1, columns=self.betas.columns, index=["const"]), self.betas], axis=0)
            # Add a row and columns of zeros for the varianca of the intercept
            self.Omega = np.vstack((np.zeros((1, self.Omega.shape[1])), self.Omega))
            self.Omega = np.hstack((np.zeros((self.Omega.shape[0], 1)), self.Omega))

        X = self.betas.values.T
        Lhat = inv(X.T @ X) @ X.T @ self.avg_ret
        self.lambdas = pd.Series(
            data=Lhat,
            index=self.betas.index,
        )
        self.alphas = pd.Series(
            data=self.avg_ret - X @ Lhat,
            index=assets.columns,
        )

        # Conventional OLS Estimator Covariance Matrix
        # Equations 12.12 and 12.13 of Cochrane (2009)
        self.conv_cov_lambda_hat = pd.DataFrame(
            data=(1 / self.T) * (inv(X.T @ X) @ X.T @ self.Sigma @ X @ inv(X.T @ X) + self.Omega),
            index=self.lambdas.index,
            columns=self.lambdas.index,
        )
        self.conv_cov_alpha_hat = pd.DataFrame(
            data=(1 / self.T) * (np.eye(self.N) - X @ inv(X.T @ X) @ X.T) @ self.Sigma @ (np.eye(self.N) - X @ inv(X.T @ X) @ X.T).T,
            index=assets.columns,
            columns=assets.columns,
        )

        # Shaken Correction for Covariance Matrices
        # Equations 12.19 and 12.20 of Cochrane (2009)
        if cs_const:
            aux_covf = self.Omega.copy()[1:, 1:]
            lhat = self.lambdas.copy().iloc[1:].values
        else:
            aux_covf = self.Omega.copy()
            lhat = self.lambdas.copy().values

        self.shanken_factor = 1 + lhat.T @ inv(aux_covf) @ lhat

        self.shanken_cov_lambda_hat = pd.DataFrame(
            data=(1 / self.T) * (self.shanken_factor * inv(X.T @ X) @ X.T @ self.Sigma @ X @ inv(X.T @ X) + self.Omega),
            index=self.lambdas.index,
            columns=self.lambdas.index,
        )
        self.shanken_cov_alpha_hat = self.conv_cov_alpha_hat * self.shanken_factor

class RiskPremiaTermStructure:
    """
    This class implements the unconditional version of the model presented in

        Bryzgalova, Svetlana and Huang, Jiantao and Julliard, Christian,
        Macro Strikes Back: Term Structure of Risk Premia (March 8, 2024).
        Available at SSRN: https://ssrn.com/abstract=4752696

    The model identifies the shocks common to financial markets and the
    factor (tradeable or not, potentially predictable), their propagation across
    horizons, and the term structure of risk premia.
    """
    k_max = 15

    def __init__(
            self,
            assets,
            factor,
            s_bar,
            n_draws=1000,
            burnin=1000,
            k=None,
    ):
        """
        Parameters
        ----------
        assets: pandas.DataFrame
            Timeseries of asset returns.

        factor: pandas.Series
            Factor from which to identify premia. Can be a tradable or
            non-tradable factor.

        s_bar: int
            Number of lags used in the MA representation of the factor

        n_draws: int
            Number of draws (after the burnin) from the Gibbs sampling procedure

        burnin: int
            Number of beggining draws to be dropped from the analysis.

        k: int
            Number of common factors in the model. If None, selects the number
            automatically based on information criteria
        """
        self._assertions(assets, factor)

        # Simple attributes
        self.assets = assets
        self.macro_factor = factor
        self.t, self.n = assets.shape
        self.s_bar = s_bar
        self.n_draws = n_draws
        self.burnin = burnin

        # select number of latent factors
        if k is None:
            self.k = self._get_number_latent_factors()
        else:
            assert isinstance(k, int), "`k` must be an integer"
            self.k = k

        self.draws_lambda_g = self._run_unconditional_gibbs()

    def plot_premia_term_structure(self, size=5):
        # TODO Documentation
        # TODO add CI

        fig = plt.figure(figsize=(size * (16 / 7.3), size))

        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.plot(self.draws_lambda_g.median(), label="median", color="tab:blue")
        ax.fill_between(
            self.draws_lambda_g.columns,
            self.draws_lambda_g.quantile(0.05),
            self.draws_lambda_g.quantile(0.95),
            label="90% Credible Interval",
            color="tab:blue",
            alpha=0.2,
            lw=0,
        )
        ax.axhline(0, color='black', lw=0.5)
        ax.set(title=r"$\lambda_{g}^{S}$", xlabel=r"$S$")
        ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="upper left")

        plt.tight_layout()
        # TODO save fig
        plt.show()

    def _run_unconditional_gibbs(self):

        # Auxiliar matrices that do NOT update every draw
        R = self.assets.values
        mu_r = self.assets.mean().values.reshape(-1, 1)
        G = self.macro_factor.values[self.s_bar:].reshape(-1, 1)
        mu_g = self.macro_factor.mean()
        G_bar = (G - mu_g)
        D_r = np.eye(self.k + 1)
        D_r[0, 0] = 0

        # PCs of Returns
        pca = PCA(n_components=self.k)
        pca.fit(self.assets)
        V0 = pca.fit_transform(self.assets)
        windows = sliding_window_view(V0, window_shape=self.s_bar + 1, axis=0)
        windows = windows[:, ::-1, :]
        V0_mat = windows.reshape(self.t - self.s_bar, (self.s_bar + 1) * self.k)
        X = sm.add_constant(V0_mat)  # adds intercept as first column
        model = sm.OLS(self.macro_factor.iloc[self.s_bar:].to_numpy(), X).fit()
        beta_vec = model.params[1:]
        beta_mat = beta_vec.reshape(self.k, self.s_bar + 1)
        # the best rank-1 approximation direction for the columns of beta_mat, summarizes the dominant pattern in beta loadings
        U, s, Vt = svd(beta_mat, full_matrices=False)

        # Starting draw
        ups = V0.T  # latent factors
        mu_ups = ups.mean(axis=1).reshape(self.k, -1)
        rho_g = Vt[0, :].reshape(-1, 1) * s[0]
        rho_g = np.insert(rho_g, 0, mu_g).reshape(-1,1)
        eta_g = U[:, [0]]
        B_r = np.zeros((self.k + 1, self.n))

        # Dataframe to save the draws
        draws_lambda_g = pd.DataFrame(columns=range(self.s_bar + 1))
        for dd in tqdm(range(self.n_draws + self.burnin)):
            # ----- STEP 1 -----
            V_rho = self._build_V_rho(ups, mu_ups, eta_g, self.s_bar, self.t)

            # Draw of \sigma^2_{wg}
            s2_wg = invgamma.rvs(
                0.5 * (self.t - self.s_bar),
                scale=(0.5 * (G - V_rho @ rho_g).T @ (G - V_rho @ rho_g))[0, 0],
            )
            # TODO save this draw?

            # Draw of rho_g
            rho_g_hat = inv(V_rho.T @ V_rho) @ V_rho.T @ G  # arg1
            wg_hat = G - V_rho @ rho_g_hat
            Sigma_hat_rho = self._build_Sigma_hat(V_rho, self.t, self.s_bar, wg_hat)
            rho_g = multivariate_normal.rvs(mean=rho_g_hat.reshape(-1), cov=Sigma_hat_rho)
            # TODO save this draw?

            # Draw of eta_g
            V_eta = self._build_V_eta(self.t, self.s_bar, self.k, rho_g[1:], ups, mu_ups)
            eta_g_hat = inv(V_eta.T @ V_eta) @ V_eta.T @ G_bar  # arg1
            weta_hat = G_bar - V_eta @ eta_g_hat
            Sigma_hat_eta = self._build_Sigma_hat(V_eta, self.t, self.s_bar, weta_hat)
            eta_g = multivariate_normal.rvs(
                mean=eta_g_hat.reshape(-1),
                cov=Sigma_hat_eta,
            ).reshape(-1, 1)
            eta_g = eta_g / np.sqrt(eta_g.T @ eta_g)  # Normalize
            # TODO save this draw?

            # ----- STEP 2 -----
            V_r = np.column_stack([np.ones(self.t), (ups.T - mu_ups.T)])

            # Draw of \Sigma_{wr}
            # Sigma_wr = invwishart.rvs(  # TODO only for low dimension
            #     df=self.t,
            #     scale=(R - V_r @ B_r).T @ (R - V_r @ B_r),
            # )

            Sigma_wr = np.diag(
                invgamma.rvs(
                    self.t - self.k - 1,
                    scale=np.diag((1 / self.t) * (R - V_r @ B_r).T @ (R - V_r @ B_r)),
                )
            )

            # Draw of B_r
            A = V_r.T @ V_r + D_r
            B_r = matrix_normal.rvs(
                mean=np.linalg.solve(A, V_r.T @ R),  # Efficient computation
                rowcov=inv(A),
                colcov=Sigma_wr,
            )
            # TODO save draw?

            # ----- STEP 3 -----
            # Draw of \upsilon
            beta_ups = B_r.T[:, 1:]
            means = inv(beta_ups.T @ inv(Sigma_wr) @ beta_ups) @ (beta_ups.T @ inv(Sigma_wr) @ (R.T - mu_r + beta_ups @ mu_ups))
            cov = inv(beta_ups.T @ inv(Sigma_wr) @ beta_ups)
            L = cholesky(cov, lower=True)
            Z = norm.rvs(size=means.shape)
            ups = means + L @ Z
            # TODO save draw?

            # Draw of \Sigma_{upsilon}
            ups_bar = ups.mean(axis=1).reshape(-1, 1)
            Sigma_ups = invwishart.rvs(
                df=self.t - 1,
                scale=ups @ ups.T - self.t * ups_bar @ ups_bar.T,
            )
            # TODO save draw?

            mu_ups = multivariate_normal.rvs(
                mean=ups_bar.reshape(-1),
                cov=(1 / self.t) * Sigma_ups,
            ).reshape(-1, 1)
            # TODO save draw?

            # ----- STEP 4 -----
            if self.k == 1:
                Sigma_ups = np.array([[Sigma_ups]])

            Sigma_r = beta_ups @ Sigma_ups @ beta_ups.T + Sigma_wr
            mu_tilde = mu_r + 0.5 * np.diag(Sigma_r).reshape(-1, 1)
            lambda_ups = inv(beta_ups.T @ beta_ups) @ beta_ups.T @ mu_tilde

            rho = rho_g[1:]

            # save the draws of lambda_g_s
            draws_lambda_g.loc[dd] = (eta_g.T @ lambda_ups)[0, 0] * pd.Series([np.mean(np.cumsum(rho[:S + 1])) for S in range(self.s_bar + 1)])

        draws_lambda_g = draws_lambda_g.iloc[-self.n_draws:]
        return draws_lambda_g

    @staticmethod
    def _build_V_eta(T, Sbar, K, rho, v, mu_v):
        v_c = v - mu_v
        V_eta = np.empty((T - Sbar, K))
        for t in range(T - Sbar):
            V_eta[t, :] = np.sum(rho[:, None] * v_c[:, Sbar + t - np.arange(Sbar + 1)].T, axis=0)

        return V_eta

    def _build_Sigma_hat(self, V, T, Sbar, w_hat):
        sum1 = sum(
            (w_hat[it, 0] ** 2) * V[[it], :].T @ V[[it], :]
            for it in range(w_hat.shape[0])
        )
        sum2 = sum(
            self._build_Gamma_hat_rho_l(T, Sbar, w_hat, V, l) * (1 - (l / (1 + Sbar)))
            for l in range(1, Sbar + 1)
        )
        S_hat_rho = (1 / (T - Sbar)) * sum1 + sum2

        Sigma_hat_rho = inv(V.T @ V) @ ((T - Sbar) * S_hat_rho) @ inv(V.T @ V)
        return Sigma_hat_rho

    @staticmethod
    def _build_V_rho(ups, mu_ups, eta_g, sbar, t):
        elements = (ups - mu_ups).T.dot(eta_g)
        W = sliding_window_view(elements.reshape(-1), sbar + 1)
        W = W[:, ::-1]  # Reverse column order
        V_rho = np.concatenate([np.ones((t - sbar, 1)), W], axis=1)
        return V_rho

    @staticmethod
    def _build_Gamma_hat_rho_l(T, Sbar, wg_hat, V_rho, l):
        Gamma = np.zeros((V_rho.shape[1], V_rho.shape[1]))
        for it in range(l, T - Sbar):
            scalar = wg_hat[it, 0] * wg_hat[it - l, 0]
            mat1 = V_rho[[it], :].T @ V_rho[[it - l], :]
            mat2 = V_rho[[it - l], :].T @ V_rho[[it], :]
            Gamma += scalar * (mat1 + mat2)
        Gamma = Gamma * (1 / (T - Sbar - l))
        return Gamma

    def _get_number_latent_factors(self):
        retp_ret = ((self.assets - self.assets.mean()).T @ (self.assets - self.assets.mean())).values
        eigv = np.sort(eigvals(retp_ret))[::-1]
        gamma_hat = np.median(eigv[:self.k_max])
        phi_nt = 0.5 * gamma_hat * np.log(self.t * self.n) * (self.t**(-0.5) + self.n**(-0.5))
        j = (np.arange(len(eigv)) + 1)
        grid = (eigv / (self.t * self.n)) + j * phi_nt
        k_hat  = np.argmin(grid) + 1
        print("selected number of factors is", k_hat)
        return k_hat

    @staticmethod
    def _assertions(assets, factor):
        assert factor.index.equals(assets.index), \
            "the index for `factor` and `assets` must match"
