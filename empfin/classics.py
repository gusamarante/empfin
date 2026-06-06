import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import inv, solve, det
from scipy.stats import (
    chi2,
    f,
    t as t_dist,
)
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from empfin.utils import nearest_psd


# TODO models to implement
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


class FamaMacBeth:
    """
    References:
        Fama, Eugene F., and James D. MacBeth (1973)
        "Risk, Return, and Equilibrium: Empirical Tests"
        Journal of Political Economy, 81(3), 607-636

        Cochrane, John (2005)
        "Asset Pricing: Revised Edition"
        Chapter 12

        Shanken, Jay (1992)
        "On the Estimation of Beta-Pricing Models"
        Review of Financial Studies, 5(1), 1-33

        Petersen, Mitchell A. (2009)
        "Estimating Standard Errors in Finance Panel Data Sets:
        Comparing Approaches"
        Review of Financial Studies, 22(1), 435-480
    """

    def __init__(self, assets, factors, rolling_window=None, nw_lags=None, shanken=False):
        """
        Two-pass Fama-MacBeth (1973) estimator of factor risk premia.

        First pass: for each asset, run a time-series regression of its excess
        return on the factors with an intercept; the slope coefficients are
        that asset's betas (factor loadings).

        Second pass: at each period t, run a cross-sectional OLS regression of
        that period's asset returns on the first-pass betas, with an intercept.
        The risk premium estimate for each factor is the time-series mean of
        the cross-sectional slopes; standard errors come from the time-series
        variability of those slopes.

        Defaults are the classical, unconditional, full-sample Fama-MacBeth
        with no errors-in-variables correction.

        Parameters
        ----------
        assets: pandas.DataFrame
            Timeseries of test asset excess returns, shape (T, N).

        factors: pandas.DataFrame, pandas.Series
            Timeseries of factor returns, shape (T, K). Index must match
            `assets.index`.

        rolling_window: int, optional
            If None (default), first-pass betas are estimated once over the
            full sample. If an integer `w`, betas are re-estimated at each
            t over the window `[t-w+1, t]` (inclusive, ending at t -- no
            future data is used). The cross-section at time t then uses these
            time-varying betas.

        nw_lags: int, optional
            If None (default), standard errors are the classical Fama-MacBeth
            standard errors -- the time-series std dev (ddof=1) of the
            cross-sectional slopes divided by sqrt(T_eff). If a non-negative
            integer L, Newey-West HAC standard errors are computed on the
            time series of cross-sectional slopes using L as the truncation
            lag (Bartlett kernel).

        shanken: bool
            If False (default), no errors-in-variables correction is applied.
            If True, all standard errors are multiplied by sqrt(1 + lambda'
            Sigma_f^{-1} lambda), where lambda excludes the intercept and
            Sigma_f is the sample covariance of the factors. The correction
            composes with the chosen base variance (FM or Newey-West).

        Attributes
        ----------
        T: int
            Length of the input sample.

        N: int
            Number of test assets.

        K: int
            Number of factors.

        T_eff: int
            Number of cross-sections that were actually fit (T if
            `rolling_window is None`; otherwise T - rolling_window + 1, minus
            any periods skipped due to too few valid assets or a singular
            design matrix).

        lambdas: pandas.Series
            Risk premia estimates, indexed by ['const'] + factor names. Each
            element is the time-series mean of the corresponding
            cross-sectional slope.

        lambdas_se: pandas.Series
            Standard errors of `lambdas`. Source depends on `nw_lags` and
            `shanken`.

        lambdas_tstat: pandas.Series
            t-statistics: `lambdas / lambdas_se`.

        lambdas_pvalue: pandas.Series
            Two-sided p-values for `lambdas_tstat`, computed against the t
            distribution with `T_eff - 1` degrees of freedom.

        lambdas_t: pandas.DataFrame
            Time series of cross-sectional regression slopes, shape
            (T_eff, K+1), with columns ['const'] + factor names. This is the
            core object on which all inference is based.

        alphas_t: pandas.DataFrame
            Time series of cross-sectional regression residuals (pricing
            errors), shape (T_eff, N). Assets dropped from a particular
            cross-section appear as NaN in that row.

        betas: pandas.DataFrame
            Full-sample first-pass betas, shape (K, N). Only set when
            `rolling_window is None`.

        betas_t: pandas.DataFrame
            Rolling first-pass betas, with a MultiIndex (date, factor) on
            the rows and asset names on the columns. Only set when
            `rolling_window is not None`. Access the K x N betas at time t
            via `betas_t.loc[t]`.

        Notes
        -----
        The risk premium estimate is computed from the time series of
        cross-sectional regressions, not from a single cross-sectional
        regression on average returns. With fixed betas the two procedures
        give the same point estimate but the latter yields incorrect
        standard errors (Cochrane 2005, Ch. 12). For the convenience of
        running the second pass on mean returns, see `CrossSectionReg`.

        When `rolling_window=w`, the betas used in the cross-section at
        time t are estimated from the w observations ending at t inclusive,
        so the first w-1 cross-sections are skipped.

        Newey-West (Petersen 2009) on the lambda time series corrects for
        serial correlation in the cross-sectional slope estimates; the
        truncation lag is the user's responsibility.
        """
        if isinstance(factors, pd.Series):
            factors = factors.to_frame()
        if isinstance(assets, pd.Series):
            assets = assets.to_frame()

        assert assets.index.equals(factors.index), \
            "Indexes of `assets` and `factors` must be the same"

        self.T = assets.shape[0]
        self.N = assets.shape[1]
        self.K = factors.shape[1]
        self.ret_mean = assets.mean()

        if self.T < self.K + 2:
            raise ValueError(
                f"T={self.T} too small for K={self.K} factors; need T >= K + 2"
            )
        if self.N < self.K + 2:
            raise ValueError(
                f"N={self.N} too small for K={self.K} factors; need N >= K + 2 "
                f"for the cross-section to have positive degrees of freedom"
            )

        if rolling_window is not None:
            if not isinstance(rolling_window, (int, np.integer)) or rolling_window <= 0:
                raise ValueError(
                    f"rolling_window must be a positive integer or None; got {rolling_window}"
                )
            if rolling_window < self.K + 2:
                raise ValueError(
                    f"rolling_window={rolling_window} too small for K={self.K} factors; "
                    f"need rolling_window >= K + 2"
                )
            if self.T < rolling_window:
                raise ValueError(
                    f"T={self.T} too small for rolling_window={rolling_window}"
                )

        if nw_lags is not None:
            if not isinstance(nw_lags, (int, np.integer)) or nw_lags < 0:
                raise ValueError(
                    f"nw_lags must be a non-negative integer or None; got {nw_lags}"
                )

        # First pass
        if rolling_window is None:
            self.betas = self._full_sample_betas(assets, factors)
            beta_lookup = lambda t: self.betas
            valid_dates = assets.index
        else:
            self.betas_t = self._rolling_betas(assets, factors, rolling_window)
            beta_lookup = lambda t: self.betas_t.loc[t]
            valid_dates = assets.index[rolling_window - 1:]

        # Second pass
        self.lambdas_t, self.alphas_t = self._cross_section(assets, beta_lookup, valid_dates)
        self.T_eff = self.lambdas_t.shape[0]

        if self.T_eff < 2:
            raise RuntimeError(
                f"Only {self.T_eff} usable cross-section(s); cannot compute standard errors"
            )

        # Inference
        self.lambdas = self.lambdas_t.mean()
        self.lambdas.name = "Lambdas"

        if nw_lags is None:
            var = self.lambdas_t.var(ddof=1) / self.T_eff
        else:
            if nw_lags >= self.T_eff:
                raise ValueError(
                    f"nw_lags={nw_lags} must be less than T_eff={self.T_eff}"
                )
            var = self._newey_west_variance(self.lambdas_t, nw_lags)

        if shanken:
            Sigma_f = factors.cov().values
            try:
                Sigma_f_inv = inv(Sigma_f)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    "Shanken correction failed: factor covariance matrix is singular"
                ) from e
            lambda_K = self.lambdas.drop('const').values
            self.shanken_factor = float(1.0 + lambda_K @ Sigma_f_inv @ lambda_K)
            var = var * self.shanken_factor

        self.lambdas_se = pd.Series(np.sqrt(var.values), index=self.lambdas.index, name="SE")
        self.lambdas_tstat = pd.Series(
            data=self.lambdas.values / self.lambdas_se.values,
            index=self.lambdas.index,
            name="t-stat",
        )
        dof = self.T_eff - 1
        self.lambdas_pvalue = pd.Series(
            data=2.0 * (1.0 - t_dist.cdf(np.abs(self.lambdas_tstat.values), df=dof)),
            index=self.lambdas.index,
            name="p-value",
        )

    def _full_sample_betas(self, assets, factors):
        X = np.column_stack([np.ones(self.T), factors.values])  # T x (K+1)
        x_has_nan = np.isnan(X).any(axis=1)
        betas = np.full((self.K, self.N), np.nan)
        for j, asst in enumerate(assets.columns):
            y = assets[asst].values
            mask = (~np.isnan(y)) & (~x_has_nan)
            if mask.sum() < self.K + 2:
                raise ValueError(
                    f"Asset {asst!r} has only {mask.sum()} non-missing observations; "
                    f"need at least K + 2 = {self.K + 2}"
                )
            Xj = X[mask]
            yj = y[mask]
            try:
                coefs = solve(Xj.T @ Xj, Xj.T @ yj)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    f"Singular first-pass design for asset {asst!r}"
                ) from e
            betas[:, j] = coefs[1:]
        return pd.DataFrame(
            data=betas,
            index=factors.columns,
            columns=assets.columns,
        )

    def _rolling_betas(self, assets, factors, w):
        F = factors.values
        A = assets.values
        end_indices = range(w - 1, self.T)
        end_dates = [assets.index[t] for t in end_indices]

        blocks = []
        for t in tqdm(end_indices, total=self.T - w + 1, desc="Rolling first-pass"):
            X = np.column_stack([np.ones(w), F[t - w + 1:t + 1, :]])
            Y = A[t - w + 1:t + 1, :]
            beta_t = np.full((self.K, self.N), np.nan)
            x_mask = ~np.isnan(X).any(axis=1)
            for j in range(self.N):
                y = Y[:, j]
                mask = x_mask & ~np.isnan(y)
                if mask.sum() < self.K + 2:
                    continue
                Xj = X[mask]
                yj = y[mask]
                try:
                    coefs = solve(Xj.T @ Xj, Xj.T @ yj)
                except np.linalg.LinAlgError:
                    continue
                beta_t[:, j] = coefs[1:]
            blocks.append(beta_t)

        multi_idx = pd.MultiIndex.from_product(
            [end_dates, list(factors.columns)],
            names=[assets.index.name or "date", "factor"],
        )
        return pd.DataFrame(
            data=np.vstack(blocks),
            index=multi_idx,
            columns=assets.columns,
        )

    def _cross_section(self, assets, beta_lookup, valid_dates):
        K = self.K
        N = self.N
        asset_names = list(assets.columns)
        param_names = ["const"] + list(beta_lookup(valid_dates[0]).index)

        lambdas_rows = []
        alphas_rows = []
        used_dates = []
        for t in valid_dates:
            beta_t = beta_lookup(t).values  # K x N
            r_t = assets.loc[t].values  # N
            mask = (~np.isnan(r_t)) & (~np.isnan(beta_t).any(axis=0))
            if mask.sum() < K + 2:
                warnings.warn(
                    f"Skipping cross-section at {t}: only {int(mask.sum())} valid asset(s); "
                    f"need at least K + 2 = {K + 2}",
                    RuntimeWarning,
                )
                continue
            r_valid = r_t[mask]
            b_valid = beta_t[:, mask]
            X = np.column_stack([np.ones(mask.sum()), b_valid.T])  # n_valid x (K+1)
            try:
                coefs = solve(X.T @ X, X.T @ r_valid)
            except np.linalg.LinAlgError:
                warnings.warn(
                    f"Skipping cross-section at {t}: singular design matrix",
                    RuntimeWarning,
                )
                continue
            resid_full = np.full(N, np.nan)
            resid_full[mask] = r_valid - X @ coefs
            lambdas_rows.append(coefs)
            alphas_rows.append(resid_full)
            used_dates.append(t)

        if not used_dates:
            raise RuntimeError("No usable cross-sections; cannot estimate risk premia")

        idx = pd.Index(used_dates, name=assets.index.name)
        lambdas_t = pd.DataFrame(
            data=np.array(lambdas_rows),
            index=idx,
            columns=param_names,
        )
        alphas_t = pd.DataFrame(
            data=np.array(alphas_rows),
            index=idx,
            columns=asset_names,
        )
        return lambdas_t, alphas_t

    @staticmethod
    def _newey_west_variance(lambdas_t, L):
        """
        Newey-West (Bartlett-kernel) HAC variance of the sample mean for
        each column of `lambdas_t`. Returns a pandas.Series of variances.
        """
        T_eff = lambdas_t.shape[0]
        x = lambdas_t.values - lambdas_t.values.mean(axis=0, keepdims=True)
        gamma0 = (x * x).sum(axis=0) / T_eff
        s2 = gamma0.copy()
        for l in range(1, L + 1):
            w_l = 1.0 - l / (L + 1)
            cov_l = (x[l:] * x[:-l]).sum(axis=0) / T_eff
            s2 = s2 + 2.0 * w_l * cov_l
        return pd.Series(s2 / T_eff, index=lambdas_t.columns)

    def _average_betas(self):
        """
        K x N DataFrame of betas for plotting/prediction. For full-sample
        FM this is just `self.betas`; for rolling FM it is the time-average
        of `self.betas_t` across dates, restored to the canonical factor
        order from `self.lambdas`.
        """
        if hasattr(self, "betas"):
            return self.betas
        factor_order = self.lambdas.drop("const").index
        return self.betas_t.groupby(level="factor").mean().reindex(factor_order)

    def plot_alpha_pred(
            self,
            size=6,
            title=None,
            save_path=None,
            color1="tab:blue",
            color2="tab:orange",
    ):
        """
        Plots the time-averaged pricing errors and the risk premia together
        with their 95% confidence intervals, and compares the predicted
        average return with the realized average returns.

        Time-averaged alphas use the cross-section by cross-section
        residuals: alpha_i = mean_t(alphas_t[t, i]) with FM-style standard
        error std_t(alphas_t[t, i]) / sqrt(T_eff_i). Lambdas use the
        standard errors already in `lambdas_se` (FM, Newey-West, or
        Shanken-corrected, depending on construction). For rolling FM, the
        scatter uses time-averaged betas.

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

        # Time-averaged alphas with their CIs
        alpha_mean = self.alphas_t.mean()
        alpha_se = self.alphas_t.std(ddof=1) / np.sqrt(self.alphas_t.count())

        ax = plt.subplot2grid((2, 2), (0, 0))
        ax.set_title(r"$\alpha$ and their CI")
        ax = alpha_mean.plot(kind='bar', ax=ax, width=0.9, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            alpha_mean.values,
            yerr=alpha_se.values * 1.96,
            ls='none',
            ecolor=color2,
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # Lambdas (excluding the intercept) with their CIs
        lambdas = self.lambdas.drop('const')
        lambdas_se = self.lambdas_se.drop('const')

        ax = plt.subplot2grid((2, 2), (1, 0))
        ax.set_title(r"$\lambda$ and their CI")
        ax = lambdas.plot(kind='bar', ax=ax, width=0.9, color=color1)
        ax.axhline(0, color="black", lw=0.5)
        ax.errorbar(
            ax.get_xticks(),
            lambdas.values,
            yerr=lambdas_se.values * 1.96,
            ls='none',
            ecolor=color2,
        )
        ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
        ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        # Predicted VS actual average returns
        ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        betas_for_pred = self._average_betas()
        predicted = self.lambdas['const'] + betas_for_pred.T @ lambdas

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

    def plot_lambdas_timeseries(
            self,
            size=4,
            title=None,
            save_path=None,
            color1="tab:blue",
            color2="tab:orange",
    ):
        """
        Plots the time series of cross-sectional regression slopes
        `lambda_t` (one panel per coefficient, including the intercept),
        overlaid with the Fama-MacBeth point estimate as a horizontal
        reference. The panel title reports the point estimate and t-stat.

        Parameters
        ----------
        size: float
            Relative subplot size

        title: str, optional
            Overall figure title

        save_path: str, Path
            File path to save the picture. File type extension must be included
            (.png, .pdf, ...)

        color1: str
            Color of the lambda_t time series

        color2: str
            Color of the Fama-MacBeth mean reference line
        """
        cols = list(self.lambdas_t.columns)
        n = len(cols)
        fig, axes = plt.subplots(
            n,
            1,
            figsize=(size * (16 / 7.3), size * n / 1.6),
            sharex=True,
        )
        if n == 1:
            axes = [axes]
        if title is not None:
            fig.suptitle(title)

        for ax, name in zip(axes, cols):
            series = self.lambdas_t[name]
            mean_val = self.lambdas[name]
            t_val = self.lambdas_tstat[name]
            ax.plot(
                series.index,
                series.values,
                color=color1,
                lw=0.8,
                label=r"$\lambda_t$",
            )
            ax.axhline(
                mean_val,
                color=color2,
                ls="--",
                lw=1.0,
                label=r"$\hat{\lambda}$",
            )
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(rf"{name}: $\hat{{\lambda}}$ = {mean_val:.3f} (t = {t_val:.2f})")
            ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
            ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        axes[0].legend(frameon=True, loc="best")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()