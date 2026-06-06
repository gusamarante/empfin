import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import inv, solve, det
from scipy.stats import (
    chi2,
    f,
)
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from empfin.utils import nearest_psd


# TODO models to implement
#  Fama-Macbeth
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

        # 2nd stage - cross-sectional regression
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