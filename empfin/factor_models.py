import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.linalg import eigvals, inv, svd, solve, det
from scipy.linalg import cholesky
from scipy.stats import (
    chi2,
    f,
    invgamma,
    invwishart,
    matrix_normal,
    multivariate_normal,
    norm,
)
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from itertools import product
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm
import warnings
from empfin.utils import nearest_psd


# TODO models to implement
#  Fama-Macbeth
#  Bayesian Fama-MacBeth (move from bayesfm)
#  GMM
#  GLS


class TimeseriesReg:
    """
    References:
        Cochrane, John. (2005)
        "Asset Pricing: Revised Edition"
        Section 12.1

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

        params_se: pandas.DataFrame
            Standard error of the regression coefficients

        tstats: pandas.DataFrame
            t-statistic for the regression coefficients

        pvalues: pandas.DataFrame
            p-values for the regression coefficients

        cov_beta: dict
            A dictionary with the covariance matrix of the OLS estimator for
            the regression of each asset

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
            tstats.append(res.tvalues.rename(asst))
            pvalues.append(res.pvalues.rename(asst))
            resids.append(res.resid.rename(asst))

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
        Equation 12.6 from Cochrane (2005)
        """
        f1 = (self.T - self.N - self.K) / self.N
        f2 = 1 / (1 + self.lambdas.T @ inv(self.Omega) @ self.lambdas)
        alpha = self.params.loc['alpha']
        f3 = alpha.T @ inv(self.Sigma) @ alpha
        grs = f1 * f2 * f3
        pvalue = 1 - f.cdf(grs, dfn=self.N, dfd=self.T - self.N - self.K)
        return grs, pvalue

    def plot_alpha_pred(
            self,
            size=6,
            title=None,
            save_path=None,
            color1="tab:blue",
            color2="tab:orange",
    ):
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

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)

        color1: str
            Primary color of the chart

        color2: str
            Secondary color of the chart
        """
        plt.figure(figsize=(size * (16 / 7.3), size))
        if title is not None:
            plt.suptitle(title)

        # Alphas and their CIs
        ax = plt.subplot2grid((2, 2), (0, 0))
        ax.set_title(r"$\alpha$ and CI")
        ax = self.params.loc['alpha'].plot(kind='bar', ax=ax, width=0.9, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            self.params.loc['alpha'].values,
            yerr=self.params_se.loc['alpha'].values * 1.96,
            ls='none',
            ecolor=color2,
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # lambdas and their CIs
        ax = plt.subplot2grid((2, 2), (1, 0))
        ax.set_title(r"$\lambda$ and CI")
        ax = self.lambdas.plot(kind='bar', ax=ax, width=0.9, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            self.lambdas.values,
            yerr=np.sqrt(np.diag(self.Omega.values) / self.T) * 1.96,
            ls='none',
            ecolor=color2,
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # Predicted VS actual average returns
        ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        predicted = self.params.drop('alpha').multiply(self.lambdas, axis=0).sum()

        ax.scatter(predicted, self.ret_mean, label="Test Assets", color=color1)
        ax.axline((0, 0), (1, 1), color=color2, ls="--", label="45 Degree Line")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel(r"Predicted Average Return $\beta_i^{\prime} \lambda$")
        ax.set_ylabel(r"Realized Average Return $E(r_i)$")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="upper left")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()


class CrossSectionReg:
    """
    References:
        Cochrane, John.
        Asset Pricing: Revised Edition (2005)
        Chapter 12.2
    """

    def __init__(self, assets, factors, cs_const=False):
        """
        This model can be used whether the factor are tradeable or not.

        First, estimates betas with a linear model using time series regressions

            r_i = a_i + beta_i * f + eps_i

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
        self.ret_mean = assets.mean()
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

        self.conv_cov_lambda_hat = pd.DataFrame(
            data=(1 / self.T) * (inv(b.T @ b) @ b.T @ S @ b @ inv(b.T @ b) + O),
            index=self.lambdas.index.drop('const', errors='ignore'),
            columns=self.lambdas.index.drop('const', errors='ignore'),
        )
        self.conv_cov_alpha_hat = pd.DataFrame(
            data=nearest_psd(  # Sometimes necessary due to numerical errors
                (1 / self.T) * ((np.eye(self.N) - b @ inv(b.T @ b) @ b.T) @ S @ (np.eye(self.N) - b @ inv(b.T @ b) @ b.T).T)
            ),
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
        Equation 12.14 from Cochrane (2005)
        """
        grs = self.alphas.T @ inv(self.shanken_cov_alpha_hat) @ self.alphas
        dof = self.N - self.K
        pvalue = 1 - chi2.cdf(grs, dof)
        return grs, pvalue

    def plot_alpha_pred(
            self,
            size=6,
            title=None,
            save_path=None,
            color1="tab:blue",
            color2="tab:orange",
    ):
        """
        Plots the alphas and lambdas together with their confidence intervals,
        and compares the average return predicted by the model with the
        realized average returns.

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
            Primary color of the chart

        color2: str
            Secondary color of the chart
        """
        plt.figure(figsize=(size * (16 / 7.3), size))
        if title is not None:
            plt.suptitle(title)

        # Alphas
        ax = plt.subplot2grid((2, 2), (0, 0))
        ax.set_title(r"$\alpha$ and their CI")
        ax = self.alphas.plot(kind='bar', ax=ax, width=0.9, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            self.alphas.values,
            yerr=np.sqrt(np.diag(self.shanken_cov_alpha_hat)) * 1.96,
            ls='none',
            ecolor=color2,
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # lambdas
        lambdas = self.lambdas.drop("const", errors='ignore')
        ax = plt.subplot2grid((2, 2), (1, 0))
        ax.set_title(r"$\lambda$ and their CI")
        ax = lambdas.plot(kind='bar', ax=ax, width=0.9, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            lambdas.values,
            yerr=np.sqrt(np.diag(self.shanken_cov_lambda_hat)) * 1.96,
            ls='none',
            ecolor=color2,
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # Predicted VS actual average returns
        ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        predicted = self.lambdas.get('const', default=0) + self.betas.T @ lambdas
        ax.scatter(predicted, self.ret_mean, label="Test Assets", color=color1)
        ax.axline((0, 0), (1, 1), color=color2, ls="--", label="45 Degree Line")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel(r"Predicted Average Return $\beta_i^{\prime} \lambda$ (no $\alpha$)")
        ax.set_ylabel(r"Realized Average Return $E(r_i)$")
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="upper left")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
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

    def __init__(self, assets, factors, max_iter=1000, tol=1e-8):
        """
        This model should be used when all factors are not tradable portfolios.

        The unconstrained model is given by

            r_t = a + B * f_{K,t} + eps_t

        The constrained model imposes

            a = lambda0 iota + B * (lambda_K - mu_{f,K})

        which can then be estimated by iterative maximum likelihood. For
        details, Campbell, Lo & MacKinlay (2012), equations (6.2.39)-(6.2.41)

        Parameters
        ----------
        assets: pandas.DataFrame
            timeseries of test assets returns

        factors: pandas.DataFrame
            timeseries of the factors

        max_iter: int
            maximum number of iterations allowed in the estimation of the
            constrained model

        tol: float
            convergence criteria of iterative estimator of the constrained
            model
        """
        # Align data
        common_index = assets.index.intersection(factors.index)
        assets = assets.loc[common_index]
        factors = factors.loc[common_index]

        self.T = len(common_index)
        self.K = factors.shape[1]
        self.N = assets.shape[1]
        self.Omega_hat = factors.cov()
        self.mu_f = factors.mean()

        self.B_unc, self.a_unc, self.Sigma_unc = self._estimate_unconstrained(assets, factors)
        self.B_con, self.gamma0_con, self.gamma1_con, self.Sigma_con = self._estimate_constrained(assets, factors, max_iter, tol)

        self.lambdas = pd.Series(
            data=self.mu_f.values + self.gamma1_con,
            index=factors.columns,
            name="Lambdas",
        )
        self.var_gamma0, self.var_gamma1 = self._compute_var_lambda()
        self.cov_lambdas = (1 / self.T) * self.Omega_hat + self.var_gamma1

    def lr_test(self):
        """
        Likelihood ratio statistic to test the null hypothesis that the
        constrained model is valid

        For details, Campbell, Lo & MacKinlay (2012), equations (6.2.1)
        and (6.2.42)
        """
        J = - (self.T - 0.5 * self.N - self.K - 1) * (np.log(det(self.Sigma_unc)) - np.log(det(self.Sigma_con)))
        dof = self.N - self.K - 1
        pvalue = 1 - chi2.cdf(J, dof)
        return J, pvalue

    def _estimate_unconstrained(self, assets, factors):
        Y_vals = assets.values
        X_vals = factors.values

        mu_Y = Y_vals.mean(axis=0)
        mu_X = X_vals.mean(axis=0)

        Y_c = Y_vals - mu_Y
        X_c = X_vals - mu_X

        XtX = X_c.T @ X_c
        XtY = X_c.T @ Y_c
        B_hat = solve(XtX, XtY).T

        a_hat = mu_Y - B_hat @ mu_X

        resid = Y_vals - a_hat - X_vals @ B_hat.T
        Sigma_hat = (resid.T @ resid) / self.T

        return B_hat, a_hat, Sigma_hat

    def _estimate_constrained(self, assets, factors, max_iter, tol):
        iota = np.ones((self.N, 1))

        Y_vals = assets.values
        X_vals = factors.values

        mu_Y = Y_vals.mean(axis=0)
        mu_X = X_vals.mean(axis=0)

        # Initialized with unconstrained estimates
        B = self.B_unc.copy()
        Sigma = self.Sigma_unc.copy()

        # Hold previous values for convergence check
        prev_B = B.copy()
        prev_Sigma = Sigma.copy()
        prev_gamma = np.zeros(self.K + 1)

        pbar = tqdm(range(max_iter), desc="Diff = 1")
        for _ in pbar:

            # Update Gamma
            Sigma_inv = inv(Sigma)
            X = np.hstack([iota, B])  # N x (K+1)
            target = mu_Y - B @ mu_X  # (N,)
            Xt_Sinv = X.T @ Sigma_inv
            LHS = Xt_Sinv @ X
            RHS = Xt_Sinv @ target
            gamma = solve(LHS, RHS)  # (K+1,)
            gamma_0 = gamma[0]
            gamma_1 = gamma[1:]

            # Update B
            Y_adj = Y_vals - gamma_0  # T x N (Broadcasting scalar)
            X_adj = X_vals + gamma_1  # T x K (Broadcasting vector)
            Denom = X_adj.T @ X_adj
            Num = Y_adj.T @ X_adj
            B = solve(Denom, Num.T).T

            # Update Sigma
            resid = Y_vals - gamma_0 - X_adj @ B.T  # X_adj already contains gamma_1
            Sigma = (resid.T @ resid) / self.T

            # Check convergence
            max_gamma = np.abs(gamma - prev_gamma).max()
            max_B = np.abs(B - prev_B).max()
            max_Sigma = np.abs(Sigma - prev_Sigma).max()
            max_diff = max(max_gamma, max_B, max_Sigma)

            pbar.set_description(f"Diff = {max_diff}")
            if max_diff < tol:
                break

            # Hold values for next iteration
            prev_gamma = gamma.copy()
            prev_B = B.copy()
            prev_Sigma = Sigma.copy()

        else:
            warnings.warn(f"Convergence not achieved after {max_iter} iterations", ConvergenceWarning)

        return B, gamma_0, gamma_1, Sigma

    def _compute_var_lambda(self):
        """
        Equations 6.2.44 and 6.2.45 from Campbell, Lo & McKinley (2012)
        """
        sf = (1 / self.T) * (1 + self.lambdas.T @ inv(self.Omega_hat) @ self.lambdas)
        iota = np.ones((self.N, 1))
        A = iota.T @ inv(self.Sigma_con) @ iota
        C = iota.T @ inv(self.Sigma_con) @ self.B_con
        D = inv(self.B_con.T @ inv(self.Sigma_con) @ self.B_con)

        var_g0 = sf * inv(A - C @ D @ C.T)
        var_g1 = sf * D + var_g0 * D @ C.T @ C @ D

        return var_g0, var_g1

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

        self.draws_lambda_g, self.draws_loadings, self.draws_eta_g, self.draws_rho, self.draws_Sigma_ups, self.draws_Sigma_r = self._run_unconditional_gibbs()

    def factor_mimicking_portfolio(self, S):
        # Reshape draws into 3D arrays for batched computation
        beta_ups_all = self.draws_loadings.values.reshape(self.n_draws, self.n, self.k)
        Sigma_ups_all = self.draws_Sigma_ups.values.reshape(self.n_draws, self.k, self.k)
        Sigma_r_all = self.draws_Sigma_r.values.reshape(self.n_draws, self.n, self.n)
        eta_g_all = self.draws_eta_g.values.reshape(self.n_draws, self.k, 1)

        # Batched: beta_ups @ Sigma_ups @ eta_g for all draws → (n_draws, n, 1)
        rhs = beta_ups_all @ Sigma_ups_all @ eta_g_all

        # Batched solve: Sigma_r @ w = rhs for all draws → (n_draws, n, 1)
        base_w = np.linalg.solve(Sigma_r_all, rhs)

        # S-dependent scalar per draw: sum_{l=0}^{min(s_bar, S+1)} rho_l * (S+2-l) / (S+2)
        upper = min(self.s_bar, S + 1)
        l_vals = np.arange(upper + 1)
        coeffs = S + 2 - l_vals
        scalars = self.draws_rho.values[:, :upper + 1] @ coeffs / (S + 2)

        # Scale base weights by the S-dependent scalar per draw
        w = base_w.reshape(self.n_draws, self.n) * scalars[:, None]

        return pd.DataFrame(
            data=w,
            columns=self.assets.columns,
        )

    def plot_premia_term_structure(
            self,
            ci=0.9,
            size=5,
            title=r"$\lambda_{g}^{S}$",
            x_axis_title=r"$S$",
            save_path=None,
            show_chart=True,
    ):
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

        title: str
            Title of the chart

        x_axis_title: str
            Title of the x axis

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)

        show_chart: bool
            If True, shows the chart. If False, it still saves the chart to
            `save_path`.
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
        ax.set(
            title=title,
            xlabel=x_axis_title,
        )
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.legend(frameon=True, loc="best")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()

    def plot_loadings_heatmap(self, figsize=(5 * (16 / 7.3), 5), save_path=None, show_chart=True):
        """

        Parameters
        ----------
        figsize: tuple
            Overall size of the figure

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)

        show_chart: bool
            If True, shows the chart. If False, it still saves the chart to
            `save_path`.
        """
        df = pd.DataFrame(
            data=self.draws_loadings.median().values.reshape(self.n, self.k),
            columns=[k + 1 for k in range(self.k)],
            index=self.assets.columns,
        )

        plt.figure(figsize=figsize)
        ax = plt.subplot2grid((1, 1), (0, 0))

        ax = sns.heatmap(
            data=df,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            # cbar=None,
            # linewidths=0.5,
            # annot=True,
            # linecolor="white",
        )
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set(
            title="Medians of Factor Loadings",
        )

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()

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
        rho_g = np.insert(rho_g, 0, mu_g).reshape(-1, 1)
        eta_g = U[:, [0]]
        B_r = np.zeros((self.k + 1, self.n))

        # Pre-allocate numpy arrays to save the draws
        total_draws = self.n_draws + self.burnin
        draws_lambda_g_arr = np.empty((total_draws, self.s_bar + 1))
        draws_loadings_arr = np.empty((total_draws, self.n * self.k))
        draws_eta_g_arr = np.empty((total_draws, self.k))
        draws_rho_arr = np.empty((total_draws, self.s_bar + 1))
        draws_Sigma_ups_arr = np.empty((total_draws, self.k * self.k))
        draws_Sigma_r_arr = np.empty((total_draws, self.n * self.n))

        for dd in tqdm(range(self.n_draws + self.burnin)):
            # ----- STEP 1 -----
            V_rho = self._build_V_rho(ups, mu_ups, eta_g, self.s_bar, self.t)

            # Draw of \sigma^2_{wg}
            s2_wg = invgamma.rvs(
                0.5 * (self.t - self.s_bar),
                scale=(0.5 * (G - V_rho @ rho_g).T @ (G - V_rho @ rho_g))[0, 0],
            )

            # Draw of rho_g
            rho_g_hat = inv(V_rho.T @ V_rho) @ V_rho.T @ G  # arg1
            wg_hat = G - V_rho @ rho_g_hat
            Sigma_hat_rho = self._build_Sigma_hat(V_rho, self.t, self.s_bar, wg_hat)
            rho_g = multivariate_normal.rvs(mean=rho_g_hat.reshape(-1), cov=Sigma_hat_rho)

            # Draw of eta_g
            V_eta = self._build_V_eta(self.t, self.s_bar, self.k, rho_g[1:], ups, mu_ups)
            eta_g_hat = inv(V_eta.T @ V_eta) @ V_eta.T @ G_bar  # arg1
            weta_hat = G_bar - V_eta @ eta_g_hat
            Sigma_hat_eta = self._build_Sigma_hat(V_eta, self.t, self.s_bar, weta_hat)
            eta_g = multivariate_normal.rvs(
                mean=eta_g_hat.reshape(-1),
                cov=nearest_psd(Sigma_hat_eta),
            ).reshape(-1, 1)
            eta_g = eta_g / np.sqrt(eta_g.T @ eta_g)  # Normalize
            draws_eta_g_arr[dd] = eta_g.flatten()

            # ----- STEP 2 -----
            V_r = np.column_stack([np.ones(self.t), (ups.T - mu_ups.T)])

            # Draw of \Sigma_{wr} - only for low dimension
            # Sigma_wr = invwishart.rvs(
            #     df=self.t,
            #     scale=(R - V_r @ B_r).T @ (R - V_r @ B_r),
            # )

            sigma_wr_diag = invgamma.rvs(
                self.t - self.k - 1,
                scale=np.diag((1 / self.t) * (R - V_r @ B_r).T @ (R - V_r @ B_r)),
            )
            Sigma_wr = np.diag(sigma_wr_diag)

            # Draw of B_r
            A = V_r.T @ V_r + D_r
            B_r = matrix_normal.rvs(
                mean=np.linalg.solve(A, V_r.T @ R),  # Efficient computation
                rowcov=nearest_psd(inv(A)),
                colcov=nearest_psd(Sigma_wr),
            )

            # ----- STEP 3 -----
            # Draw of \upsilon
            beta_ups = B_r.T[:, 1:]
            draws_loadings_arr[dd] = beta_ups.flatten()

            # My previous attempt
            # means = inv(beta_ups.T @ inv(Sigma_wr) @ beta_ups) @ (beta_ups.T @ inv(Sigma_wr) @ (R.T - mu_r + beta_ups @ mu_ups))
            # cov = inv(beta_ups.T @ inv(Sigma_wr) @ beta_ups)
            # Exploit diagonal Sigma_wr: inv(Sigma_wr) @ X = X / sigma_wr_diag[:, None]
            beta_scaled = beta_ups / sigma_wr_diag[:, None]
            cov = nearest_psd(inv(beta_scaled.T @ beta_ups))
            means = cov @ (beta_scaled.T @ (R.T - mu_r + beta_ups @ mu_ups))
            L = cholesky(cov, lower=True)
            Z = norm.rvs(size=means.shape)
            ups = means + L @ Z

            # Draw of \Sigma_{upsilon}
            ups_bar = ups.mean(axis=1).reshape(-1, 1)
            Sigma_ups = invwishart.rvs(
                df=self.t - 1,
                scale=ups @ ups.T - self.t * ups_bar @ ups_bar.T,
            )

            mu_ups = multivariate_normal.rvs(
                mean=ups_bar.reshape(-1),
                cov=(1 / self.t) * Sigma_ups,
            ).reshape(-1, 1)

            # ----- STEP 4 -----
            if self.k == 1:
                Sigma_ups = np.array([[Sigma_ups]])

            Sigma_r = beta_ups @ Sigma_ups @ beta_ups.T + Sigma_wr
            mu_tilde = mu_r + 0.5 * np.diag(Sigma_r).reshape(-1, 1)
            lambda_ups = inv(beta_ups.T @ beta_ups) @ beta_ups.T @ mu_tilde

            rho = rho_g[1:]

            draws_rho_arr[dd] = rho.flatten()
            draws_Sigma_ups_arr[dd] = Sigma_ups.flatten()
            draws_Sigma_r_arr[dd] = Sigma_r.flatten()

            # save the draws of lambda_g_s
            rho_cumsum = np.cumsum(rho.flatten())
            draws_lambda_g_arr[dd] = (eta_g.T @ lambda_ups)[0, 0] * np.cumsum(rho_cumsum) / np.arange(1, self.s_bar + 2)

        # Convert to DataFrames after the loop, keeping only post-burnin draws
        loadings_columns = [f"{a} - loading {v + 1}" for a, v in product(self.assets.columns, range(self.k))]
        draws_lambda_g = pd.DataFrame(draws_lambda_g_arr[-self.n_draws:], columns=range(self.s_bar + 1))
        draws_loadings = pd.DataFrame(draws_loadings_arr[-self.n_draws:], columns=loadings_columns)
        draws_eta_g = pd.DataFrame(draws_eta_g_arr[-self.n_draws:], columns=[f"eta_g_{v + 1}" for v in range(self.k)])
        draws_rho = pd.DataFrame(draws_rho_arr[-self.n_draws:], columns=[f"rho_{l}" for l in range(self.s_bar + 1)])
        Sigma_ups_columns = [f"Sigma_ups_{i + 1}_{j + 1}" for i, j in product(range(self.k), range(self.k))]
        draws_Sigma_ups = pd.DataFrame(draws_Sigma_ups_arr[-self.n_draws:], columns=Sigma_ups_columns)
        Sigma_r_columns = [f"Sigma_r_{a}_{b}" for a, b in product(self.assets.columns, self.assets.columns)]
        draws_Sigma_r = pd.DataFrame(draws_Sigma_r_arr[-self.n_draws:], columns=Sigma_r_columns)
        return draws_lambda_g, draws_loadings, draws_eta_g, draws_rho, draws_Sigma_ups, draws_Sigma_r

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
        phi_nt = 0.5 * gamma_hat * np.log(self.t * self.n) * (self.t ** (-0.5) + self.n ** (-0.5))
        j = (np.arange(len(eigv)) + 1)
        grid = (eigv / (self.t * self.n)) + j * phi_nt
        k_hat = np.argmin(grid) + 1
        print("selected number of factors is", k_hat)
        return k_hat

    @staticmethod
    def _assertions(assets, factor):
        assert factor.index.equals(assets.index), \
            "the index for `factor` and `assets` must match"
