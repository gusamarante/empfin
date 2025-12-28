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


# TODO models to implement
#  Non-tradable factor (CLM)
#  Cross-sectional regression
#  Fama-Macbeth
#  GMM
#  GLS
#  Move bayesfm here as well


class TimeseriesReg:
    """
    References:
        Cochrane, John. (2009)
        "Asset Pricing: Revised Edition"
        Section 12.1

        Jensen, Michael C. and Black, Fischer and Scholes, Myron S. (1972)
        "The Capital Asset Pricing Model: Some Empirical Tests"
        STUDIES IN THE THEORY OF CAPITAL MARKETS, Praeger Publishers Inc.
        Available at SSRN: https://ssrn.com/abstract=908569

        Campbell, John Y., Andrew W. Lo, and Archie Craig MacKinlay (2012)
        "The Econometrics of Financial Markets"
        Princeton University Press
        Section 6.2.1
    """

    def __init__(self, assets, factors):
        """
        Estimates a linear factor model for each asset using time series
        regressions.

            r_it = alpha_i + beta_i * ft + eps_it

        All test assets r and factors f must be excess returns.

        Parameters
        ----------
        assets: pandas.DataFrame
            timeseries of test assets returns

        factors: pandas.DataFrame, pandas.Series
            timeseries of excess returns of factor portfolios

        Attributes
        ----------
        T: int
            timeseries sample size

        N: int
            number of test assets

        K: int
            number of factors

        lambdas: pandas.Series
            Risk premia estimates, which in the timeseries regression is simply
            the historical average of returns of the factor portfolios

        Omega: pandas.DataFrame
            Covariance matrix of the factor portfolios

        params: pandas.DataFrame
            Estimated regression coefficients

        cov_beta: dict
            A dictionary with the assets as keys and their covariance matrix of
            the OLS estimator as the values

        resids: pandas.DataFrame
            Timeseries of the residuals for each of the rergessions

        Sigma: pandas.DataFrame
            Covariance matrix of the residuals of each regression
        """

        assert assets.index.equals(factors.index), \
            "Indexes of `assets` and `factors` must be the same"

        if isinstance(assets, pd.Series):
            assets = assets.to_frame()

        if isinstance(factors, pd.Series):
            factors = factors.to_frame()

        self.T = assets.shape[0]  # timeseries sample size
        self.N = assets.shape[1]  # number of test assets
        self.K = factors.shape[1]  # number of factors

        self.ret_mean = assets.mean()
        self.lambdas = factors.mean()  # Risk premia estimates are their historical means
        self.Omega = factors.cov()

        # OLS for each asset
        params = []
        params_se = []
        resids = []
        tstats = []
        pvalues = []
        self.cov_beta = dict()
        for asst in assets.columns:
            model = sm.OLS(assets[asst], sm.add_constant(factors))
            res = model.fit()

            params.append(res.params.rename(asst))
            params_se.append(res.bse.rename(asst))
            resids.append(res.resid.rename(asst))
            tstats.append(res.tvalues.rename(asst))
            pvalues.append(res.pvalues.rename(asst))

            self.cov_beta[asst] = res.cov_params()

        self.params = pd.concat(params, axis=1).rename({"const": "alpha"}, axis=0)
        self.params_se = pd.concat(params_se, axis=1).rename({"const": "alpha"}, axis=0)
        self.tstats = pd.concat(tstats, axis=1).rename({"const": "alpha"}, axis=0)
        self.pvalues = pd.concat(pvalues, axis=1).rename({"const": "alpha"}, axis=0)

        self.resids = pd.concat(resids, axis=1)
        self.Sigma = self.resids.cov()

    def grs_test(self):
        """
        Runs the Gibbons-Ross-Shanken test to evaluate if all alphas are
        jointly equal to zero.

        Returns
        -------
        grs: float
            Gibbons-Ross-Shanken test statistic

        pvalue: float
            p-value of the Gibbons-Ross-Shanken statistic

        Notes
        -----
        Equation 12.6 from Cochrane (2009)
        """
        f1 = (self.T - self.N - self.K) / self.N
        f2 = 1 / (1 + self.lambdas.T @ inv(self.Omega) @ self.lambdas)
        alpha = self.params.loc['alpha']
        f3 = alpha.T @ inv(self.Sigma) @ alpha
        grs = f1 * f2 * f3
        pvalue = 1 - f.cdf(grs, dfn=self.N, dfd=self.T - self.N - self.K)
        return grs, pvalue

    def plot_alpha_pred(self, size=6, title=None):
        """
        Plots the alphas and lambdas together with their confidence intervals,
        and compares the predicted average return with the realized average
        returns.

        Parameters
        ----------
        size: float
            Relative size of the chart

        title: str, optional
            Title for the chart
        """
        plt.figure(figsize=(size * (16 / 7.3), size))
        if title is not None:
            plt.suptitle(title)

        # Alphas and their CIs
        ax = plt.subplot2grid((2, 2), (0, 0))
        ax.set_title(r"$\alpha$ and CI")
        ax = self.params.loc['alpha'].plot(kind='bar', ax=ax, width=0.9)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            self.params.loc['alpha'].values,
            yerr=self.params_se.loc['alpha'].values * 1.96,
            ls='none',
            ecolor='tab:orange',
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # lambdas and their CIs
        ax = plt.subplot2grid((2, 2), (1, 0))
        ax.set_title(r"$\lambda$ and CI")
        ax = self.lambdas.plot(kind='bar', ax=ax, width=0.9)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            self.lambdas.values,
            yerr=np.sqrt(np.diag(self.Omega.values) / self.T) * 1.96,
            ls='none',
            ecolor='tab:orange',
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # Predicted VS actual average returns
        ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        predicted = self.params.drop('alpha').multiply(self.lambdas, axis=0).sum()

        ax.scatter(predicted, self.ret_mean, label="Test Assets")
        ax.axline((0, 0), (1, 1), color="tab:orange", ls="--", label="45 Degree Line")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel(r"Predicted Average Return $\beta_i^{\prime} \lambda$")
        ax.set_ylabel(r"Realized Average Return $E(r_i)$")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="upper left")

        plt.tight_layout()
        plt.show()
        plt.close()


class NonTradableFactors:
    """
    References:
        Campbell, John Y., Andrew W. Lo, and Archie Craig MacKinlay (2012)
        "The Econometrics of Financial Markets"
        Princeton University Press
        Section 6.2.3
    """

    def __init__(self, assets, factors):
        # TODO Documentation
        # TODO Implement
        #  Unconstrained model (Stacked)
        #  Constrained Model  (Iterated MLE)
        #  Factor risk premia  (from the model parameters)
        #  "GRS" test  (CLM 6.2.42)
        """
        def estimate_unconstrained_model(assets, factors):

            # 1. Align Data
            # Ensure we only use dates present in both datasets
            common_index = assets.index.intersection(factors.index)
            Y = assets.loc[common_index]
            X = factors.loc[common_index]

            T = len(common_index)

            # Convert to NumPy arrays for linear algebra efficiency
            Y_vals = Y.values  # T x N
            X_vals = X.values  # T x K

            # 2. Compute Means (\hat{\mu} and \hat{\mu}_{f,K})
            mu_Y = Y_vals.mean(axis=0) # Shape: (N,)
            mu_X = X_vals.mean(axis=0) # Shape: (K,)

            # 3. Center the Data
            Y_c = Y_vals - mu_Y
            X_c = X_vals - mu_X

            # 4. Estimate B (Betas)
            # Formula: B_hat = [Sum(R_c f_c')] [Sum(f_c f_c')]^-1
            # Matrix form: B_hat = (Y_c.T @ X_c) @ (X_c.T @ X_c)^-1
            # To avoid explicit inversion (slow/unstable), we solve the linear system:
            # (X_c.T @ X_c) @ B_hat.T = (X_c.T @ Y_c)

            XtX = X_c.T @ X_c  # K x K
            XtY = X_c.T @ Y_c  # K x N

            # Use linalg.solve to solve for B transposed
            try:
                B_hat_T = np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if factors are collinear
                B_hat_T = np.linalg.pinv(XtX) @ XtY

            B_hat = B_hat_T.T  # N x K

            # 5. Estimate a (Alphas)
            # Formula: a_hat = mu_Y - B_hat @ mu_X
            a_hat = mu_Y - B_hat @ mu_X  # Shape: (N,)

            # 6. Estimate Sigma (Residual Covariance)
            # Formula: Sum(epsilon @ epsilon') / T
            # Residuals = Actual - (a + B @ f)
            # Note: X_vals @ B_hat.T gives the factor component for all T
            Y_pred = a_hat + X_vals @ B_hat.T
            epsilon = Y_vals - Y_pred

            Sigma_hat = (epsilon.T @ epsilon) / T  # N x N

            # 7. Wrap results in Pandas containers
            a_out = pd.Series(a_hat, index=Y.columns, name="Alpha")
            B_out = pd.DataFrame(B_hat, index=Y.columns, columns=X.columns)
            Sigma_out = pd.DataFrame(Sigma_hat, index=Y.columns, columns=Y.columns)

            return a_out, B_out, Sigma_out
        """
        pass


class CrossSectionReg:
    """
    References:
        Cochrane, John.
        Asset Pricing: Revised Edition, 2009
        Chapter 12.2
    """

    def __init__(self, assets, factors, cs_const=False):
        """
        First, estimates betas with a linear model using time series regressions

            r_i = a_i + beta_i * f + eps_i  # TODO when can this be tradeable or not?

        Then estimate factor risk premia from a regression across assets of
        average returns on the betas

            E(r_i) = (const + ) beta_i * lambda + alpha_i

        Parameters
        ----------
        assets: pandas.DataFrame
            timeseries of test assets returns

        factors: pandas.DataFrame
            timeseries of the factors

        cs_const: bool
            If True, adds a constant to the cross-sectional regression
        """
        self.avg_ret = assets.mean()

        # First stage is the timeseries regression
        ts_reg = TimeseriesReg(assets, factors)
        self.T = ts_reg.T
        self.N = ts_reg.N
        self.K = ts_reg.K
        self.betas = ts_reg.params.drop('alpha')
        self.Sigma = ts_reg.Sigma  # Coavariance of all the residuals from all the 1st pass regressions
        self.Omega = ts_reg.Omega  # Factor Covariance

        # 2nd stage - cross sectional regression
        if cs_const:
            X = sm.add_constant(self.betas.T)
        else:
            X = self.betas.T

        model = sm.OLS(self.avg_ret, X)
        res = model.fit()
        self.lambdas = res.params
        self.alphas = res.resid

        # Conventional OLS Estimator Covariance Matrix
        # Equations 12.12 and 12.13 of Cochrane (2009)
        b = self.betas.T.values
        S = self.Sigma.values
        O = self.Omega.values

        # TODO we use no value from the 2nd stage pass to compute these SEs. Should we?
        self.conv_cov_lambda_hat = pd.DataFrame(
            data=(1 / self.T) * (inv(b.T @ b) @ b.T @ S @ b @ inv(b.T @ b) + O),
            index=self.lambdas.index.drop('const', errors='ignore'),
            columns=self.lambdas.index.drop('const', errors='ignore'),
        )
        self.conv_cov_alpha_hat = pd.DataFrame(
            data=(1 / self.T) * ((np.eye(self.N) - b @ inv(b.T @ b) @ b.T) @ S @ (np.eye(self.N) - b @ inv(b.T @ b) @ b.T).T),
            index=assets.columns,
            columns=assets.columns,
        )

        # Shaken Correction for Covariance Matrices
        # Equations 12.19 and 12.20 of Cochrane (2009)
        lhat = self.lambdas.drop('const', errors='ignore').values
        self.shanken_factor = 1 + lhat.T @ inv(O) @ lhat

        self.shanken_cov_lambda_hat = pd.DataFrame(
            data=(1 / self.T) * (self.shanken_factor * inv(b.T @ b) @ b.T @ S @ b @ inv(b.T @ b) + O),
            index=self.lambdas.index.drop('const', errors='ignore'),
            columns=self.lambdas.index.drop('const', errors='ignore'),
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
            Number of draws from the beggining to be dropped from the analysis

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

    def plot_premia_term_structure(self, ci=0.9, size=5):
        """
        Plots the unconditional risk premia term structure. The point estimate
        is the median of the posterior draws and the shaded area represents the
        `ci` credible intervals.

        Parameters
        ----------
        ci: float
            Size of the credibility intervals

        size: float
            Relative size of the chart. Aspect ratio is constant at 16 / 7.3
        """

        plt.figure(figsize=(size * (16 / 7.3), size))
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.plot(self.draws_lambda_g.median(), label="median", color="tab:blue")
        ax.fill_between(
            self.draws_lambda_g.columns,
            self.draws_lambda_g.quantile((1 - ci) / 2),
            self.draws_lambda_g.quantile((1 + ci) / 2),
            label=f"{round(100 * ci)}% Credible Interval",
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
