import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.linalg import eigvals, inv, svd, solve, det
from scipy.linalg import cholesky, solve_discrete_lyapunov
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
#  GMM
#  GLS

# ========================
# ===== The Classics =====
# ========================
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


# =============================================================
# ===== Macro Strikes Back - Bryzgalova, Huang & Julliard =====
# =============================================================
def _build_V_rho(centered, eta_g, sbar, t):
    """
    Builds the regressor matrix for sampling `rho_g` in the time-series
    equation for the macro factor. `centered` is the centered/innovation
    tensor of latent factors with shape (K, T): `(v_t - mu_v)` for the
    unconditional model and the VAR innovations `eps_vt` for the conditional
    model.
    """
    elements = centered.T.dot(eta_g)
    W = sliding_window_view(elements.reshape(-1), sbar + 1)
    W = W[:, ::-1]
    V_rho = np.concatenate([np.ones((t - sbar, 1)), W], axis=1)
    return V_rho


def _build_V_eta(T, Sbar, K, rho, centered):
    """
    Builds the regressor matrix for sampling `eta_g`. `centered` has shape
    (K, T) and follows the same convention as in `_build_V_rho`.
    """
    V_eta = np.empty((T - Sbar, K))
    for t in range(T - Sbar):
        V_eta[t, :] = np.sum(
            rho[:, None] * centered[:, Sbar + t - np.arange(Sbar + 1)].T,
            axis=0,
        )
    return V_eta


def _build_Sigma_hat(V, T, Sbar, w_hat):
    """
    Newey-West sandwich estimator for the posterior covariance of `rho_g` and
    `eta_g`, accounting for autocorrelation in the residuals up to lag `Sbar`.
    """
    sum1 = sum(
        (w_hat[it, 0] ** 2) * V[[it], :].T @ V[[it], :]
        for it in range(w_hat.shape[0])
    )
    sum2 = sum(
        _build_Gamma_hat_rho_l(T, Sbar, w_hat, V, l) * (1 - (l / (1 + Sbar)))
        for l in range(1, Sbar + 1)
    )
    S_hat_rho = (1 / (T - Sbar)) * sum1 + sum2
    Sigma_hat_rho = inv(V.T @ V) @ ((T - Sbar) * S_hat_rho) @ inv(V.T @ V)
    return Sigma_hat_rho


def _build_Gamma_hat_rho_l(T, Sbar, wg_hat, V_rho, l):
    Gamma = np.zeros((V_rho.shape[1], V_rho.shape[1]))
    for it in range(l, T - Sbar):
        scalar = wg_hat[it, 0] * wg_hat[it - l, 0]
        mat1 = V_rho[[it], :].T @ V_rho[[it - l], :]
        mat2 = V_rho[[it - l], :].T @ V_rho[[it], :]
        Gamma += scalar * (mat1 + mat2)
    Gamma = Gamma * (1 / (T - Sbar - l))
    return Gamma


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
            Title of the x-axis

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
            V_rho = _build_V_rho(ups - mu_ups, eta_g, self.s_bar, self.t)

            # Draw of \sigma^2_{wg}
            s2_wg = invgamma.rvs(
                0.5 * (self.t - self.s_bar),
                scale=(0.5 * (G - V_rho @ rho_g).T @ (G - V_rho @ rho_g))[0, 0],
            )

            # Draw of rho_g
            rho_g_hat = inv(V_rho.T @ V_rho) @ V_rho.T @ G  # arg1
            wg_hat = G - V_rho @ rho_g_hat
            Sigma_hat_rho = _build_Sigma_hat(V_rho, self.t, self.s_bar, wg_hat)
            rho_g = multivariate_normal.rvs(mean=rho_g_hat.reshape(-1), cov=Sigma_hat_rho)

            # Draw of eta_g
            V_eta = _build_V_eta(self.t, self.s_bar, self.k, rho_g[1:], ups - mu_ups)
            eta_g_hat = inv(V_eta.T @ V_eta) @ V_eta.T @ G_bar  # arg1
            weta_hat = G_bar - V_eta @ eta_g_hat
            Sigma_hat_eta = _build_Sigma_hat(V_eta, self.t, self.s_bar, weta_hat)
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
                scale=nearest_psd(ups @ ups.T - self.t * ups_bar @ ups_bar.T),
            )

            mu_ups = multivariate_normal.rvs(
                mean=ups_bar.reshape(-1),
                cov=nearest_psd((1 / self.t) * Sigma_ups),
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

    def _get_number_latent_factors(self):
        retp_ret = ((self.assets - self.assets.mean()).T @ (self.assets - self.assets.mean())).values
        eigv = np.sort(eigvals(retp_ret).real)[::-1]
        eigv_normalized = eigv / (self.t * self.n)
        gamma_hat = np.median(eigv_normalized[:self.k_max])
        phi_nt = 0.5 * gamma_hat * np.log(self.t * self.n) * (self.t ** (-0.5) + self.n ** (-0.5))
        j = np.arange(1, self.k_max + 1)
        grid = eigv_normalized[:self.k_max] + j * phi_nt
        k_hat = max(1, np.argmin(grid))
        print("selected number of factors is", k_hat)
        return k_hat

    @staticmethod
    def _assertions(assets, factor):
        assert factor.index.equals(assets.index), \
            "the index for `factor` and `assets` must match"


class ConditionalRiskPremiaTermStructure:
    """
    This class implements the conditional (time-varying) version of the model
    presented in

        Bryzgalova, Svetlana and Huang, Jiantao and Julliard, Christian,
        Macro Strikes Back: Term Structure of Risk Premia.
        Available at SSRN: https://ssrn.com/abstract=4752696

    The conditional model adds a VAR(q) layer on the latent factors (optionally
    augmented with external predictors), letting the conditional mean of the
    latent factors vary over time. As a result, the term structure of risk
    premia becomes time-varying.
    """
    k_max = 15

    def __init__(
            self,
            assets,
            factor,
            s_bar,
            predictors,
            n_draws=1000,
            burnin=1000,
            k=None,
            q=1,
    ):
        """
        Parameters
        ----------
        assets: pandas.DataFrame
            timeseries of asset returns.

        factor: pandas.Series
            Macro factor for which to identify the term structure of risk
            premia. Can be a tradable or non-tradable factor.

        s_bar: int
            Number of lags used in the MA representation of the factor.

        predictors: pandas.DataFrame
            External predictors ``z_t``. The restricted VAR has the latent
            factors load only on lagged predictors (never on lagged factors),
            so predictors are required.

        n_draws: int
            Number of post-burnin Gibbs draws.

        burnin: int
            Number of initial draws to discard.

        k: int, optional
            Number of latent factors. If None, selected automatically using an
            information criterion.

        q: int
            Number of predictor lags used in the VAR. Default 1, matching the
            paper's empirical specification.
        """
        self._assertions(assets, factor, predictors)

        self.assets = assets
        self.macro_factor = factor
        self.predictors = predictors
        self.q = q
        self.t, self.n = assets.shape
        self.s_bar = s_bar
        self.n_draws = n_draws
        self.burnin = burnin

        if k is None:
            self.k = self._get_number_latent_factors()
        else:
            assert isinstance(k, int), "`k` must be an integer"
            self.k = k

        self.p = predictors.shape[1]

        # Time-varying lambda_g is defined on every observation date
        self.lambda_g_dates = self.assets.index

        (
            self.draws_lambda_g,
            self.draws_lambda_g_uncond,
            self.draws_loadings,
            self.draws_eta_g,
            self.draws_rho,
            self.draws_Sigma_eps_v,
            self.draws_Sigma_r,
            self.draws_Phi,
            self.draws_Sigma_eps_x,
        ) = self._run_conditional_gibbs()

    def factor_mimicking_portfolio(self, S):
        # TODO Review this
        r"""
        Horizon-specific factor-mimicking portfolio for the conditional
        model (Proposition 2 of Amarante and Soares "Macro Takes Time").

        The mimicking portfolio is the linear projection of the cumulative
        macro factor on the cumulative asset returns,

            w^S_MP = Var(r_{t-1->t+S})^{-1} Cov(r_{t-1->t+S}, g_{t-1->t+S})

        which under the model's distributional assumptions evaluates to

            w^S_MP = [β_v Ω(S) β_v' + (S+2) Σ_wr]^{-1}
                     β_v [Ξ(S) I_K + Θ(S)] η_g

        where Ω(S) is the cumulative latent-factor covariance over the
        holding window and Θ(S) captures the contribution from the
        predictable component of returns being correlated with past
        macroeconomic innovations. The portfolio depends on horizon `S`
        but not on the realized state of the economy.

        Parameters
        ----------
        S: int
            Holding-period horizon. Must lie in `[0, s_bar]`.

        Returns
        -------
        pandas.DataFrame
            Posterior draws of the mimicking-portfolio weights, shape
            `n_draws x N` (one row per Gibbs draw, one column per asset).
            Rows whose VAR draw is non-stationary are filled with NaN.
        """
        if not (0 <= S <= self.s_bar):
            raise ValueError(f"`S` must lie in [0, {self.s_bar}]")

        K = self.k
        p = self.p
        q = self.q
        s_bar = self.s_bar
        n_draws = self.n_draws
        N = self.n
        state_size = K + q * p

        # Selectors on the augmented state y_t = [v_t, z_t, z_{t-1}, ..., z_{t-q+1}].
        # E_v picks v_t; B injects the within-period innovation [eps_v_t; eps_z_t]
        # into the v_t and z_t blocks of y_t.
        E_v = np.zeros((K, state_size))
        E_v[:, :K] = np.eye(K)
        B_inj = np.zeros((state_size, K + p))
        B_inj[: K + p, :] = np.eye(K + p)
        EvB = E_v @ B_inj

        beta_v_all = self.draws_loadings.values.reshape(n_draws, N, K)
        eta_g_all = self.draws_eta_g.values.reshape(n_draws, K, 1)
        rho_all = self.draws_rho.values  # (n_draws, s_bar+1) -> rho_0..rho_{s_bar}
        Sigma_eps_v_all = self.draws_Sigma_eps_v.values.reshape(n_draws, K, K)
        Sigma_r_all = self.draws_Sigma_r.values.reshape(n_draws, N, N)
        Phi_all = self.draws_Phi.values.reshape(n_draws, 1 + q * p, K + p)
        Sigma_eps_x_all = self.draws_Sigma_eps_x.values.reshape(n_draws, K + p, K + p)

        # Ξ(S) — scalar, depends only on rho.
        upper = min(s_bar, S + 1)
        coeffs_xi = (S + 2 - np.arange(upper + 1)).astype(float)
        Xi_all = rho_all[:, : upper + 1] @ coeffs_xi  # (n_draws,)

        I_K = np.eye(K)
        I_state = np.eye(state_size)
        w_out = np.empty((n_draws, N))

        for dd in range(n_draws):
            beta_v = beta_v_all[dd]
            eta_g = eta_g_all[dd]
            rho = rho_all[dd]
            Sigma_eps_x = Sigma_eps_x_all[dd]
            Phi = Phi_all[dd]
            Sigma_eps_v = Sigma_eps_v_all[dd]
            Sigma_r = Sigma_r_all[dd]

            # Σ_wr is diagonal; recover it from the stored Σ_r = β_v Σ_{ε_v} β_v' + Σ_wr.
            Sigma_wr = np.diag(np.diag(Sigma_r - beta_v @ Sigma_eps_v @ beta_v.T))

            # Build the restricted-VAR transition matrix M on the augmented state.
            # The v-block depends only on lagged predictors (no self-feedback); the
            # predictor sub-block is a standard companion form on z lags.
            M_trans = np.zeros((state_size, state_size))
            for lag in range(1, q + 1):
                phi_lag = Phi[1 + (lag - 1) * p : 1 + lag * p, :]
                phi_v_lag = phi_lag[:, :K].T
                phi_z_lag = phi_lag[:, K:].T
                M_trans[:K, K + (lag - 1) * p : K + lag * p] = phi_v_lag
                M_trans[K : K + p, K + (lag - 1) * p : K + lag * p] = phi_z_lag
            for lag in range(1, q):
                M_trans[K + lag * p : K + (lag + 1) * p,
                        K + (lag - 1) * p : K + lag * p] = np.eye(p)

            # Stationarity is required for Γ_yy(0) to exist; otherwise mark NaN.
            if np.max(np.abs(eigvals(M_trans))) >= 1.0:
                w_out[dd] = np.nan
                continue

            # Γ_yy(0) from the discrete Lyapunov equation.
            Q_lyap = B_inj @ Sigma_eps_x @ B_inj.T
            try:
                Gamma_yy_0 = solve_discrete_lyapunov(M_trans, Q_lyap)
            except Exception:
                w_out[dd] = np.nan
                continue

            # Ω(S) = (S+2) γ_v(0) + Σ_{l=1}^{S+1} (S+2-l) [γ_v(l) + γ_v(l)']
            # with γ_v(l) = E_v M^l Γ_yy(0) E_v'.
            gamma_v_0 = E_v @ Gamma_yy_0 @ E_v.T
            Omega_S = (S + 2) * gamma_v_0
            M_pow = I_state.copy()
            for l in range(1, S + 2):
                M_pow = M_trans @ M_pow
                gamma_v_l = E_v @ M_pow @ Gamma_yy_0 @ E_v.T
                Omega_S = Omega_S + (S + 2 - l) * (gamma_v_l + gamma_v_l.T)

            # Ψ(h) = Cov(v_t, eps_v_{t-h}) for h = 1..s_bar.
            # Using y_t = M y_{t-1} + B eps_t, Cov(v_t, eps_v_{t-h})
            #   = E_v M^h B Σ_eps_x[:, :K] for h >= 1.
            Psi_arr = np.empty((s_bar, K, K))
            M_pow = I_state.copy()
            for h in range(1, s_bar + 1):
                M_pow = M_trans @ M_pow
                Psi_arr[h - 1] = E_v @ M_pow @ B_inj @ Sigma_eps_x[:, :K]

            # Θ(S) = Σ_{l=-(S+1)}^{S+1} (S+2-|l|) Σ_{h=max(1,1-l)}^{s_bar-max(0,l)} ρ_{h+l} Ψ(h)
            Theta_S = np.zeros((K, K))
            for l in range(-(S + 1), S + 2):
                h_lower = max(1, 1 - l)
                h_upper = s_bar - max(0, l)
                if h_lower > h_upper:
                    continue
                inner = np.zeros((K, K))
                for h in range(h_lower, h_upper + 1):
                    inner += rho[h + l] * Psi_arr[h - 1]
                Theta_S = Theta_S + (S + 2 - abs(l)) * inner

            Sigma_rr_S = beta_v @ Omega_S @ beta_v.T + (S + 2) * Sigma_wr
            Sigma_rg_S = beta_v @ (Xi_all[dd] * I_K + Theta_S) @ eta_g

            try:
                w_out[dd] = solve(Sigma_rr_S, Sigma_rg_S).flatten()
            except np.linalg.LinAlgError:
                w_out[dd] = np.nan

        return pd.DataFrame(data=w_out, columns=self.assets.columns)

    def plot_premia_time_series(
            self,
            horizons=(0, 4, 8, 12),
            ci=0.9,
            size=4,
            title=r"Time-varying term structure of risk premia $\lambda_{g,t}^{S}$",
            save_path=None,
            show_chart=True,
    ):
        """
        Plots the time-varying risk premium for each horizon in `horizons`.
        Replicates the layout of Figure 8 of the paper: each subplot shows the
        posterior median of `lambda^S_{g,t}` over time with a shaded credible
        interval.

        Parameters
        ----------
        horizons: iterable of int
            Horizons `S` (0-indexed) to plot. Each must lie in `[0, s_bar]`.
            Default `(0, 4, 8, 12)` corresponds to 1Q / 1Y / 2Y / 3Y when the
            data is quarterly.

        ci: float
            Width of the credible interval (e.g. 0.9 for 90%).

        size: float
            Relative subplot size.

        title: str
            Overall figure title.

        save_path: str, Path
            File path to save the picture (extension required).

        show_chart: bool
            Whether to display the chart after saving.
        """
        horizons = [h for h in horizons if 0 <= h <= self.s_bar]
        if len(horizons) == 0:
            raise ValueError("No valid horizons in `horizons`")

        n = len(horizons)
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

        median = np.median(self.draws_lambda_g, axis=0)
        lower = np.quantile(self.draws_lambda_g, (1 - ci) / 2, axis=0)
        upper = np.quantile(self.draws_lambda_g, (1 + ci) / 2, axis=0)

        for ax, S in zip(axes, horizons):
            ax.plot(self.lambda_g_dates, median[:, S], color="tab:blue", label="median")
            ax.fill_between(
                self.lambda_g_dates,
                lower[:, S],
                upper[:, S],
                color="tab:blue",
                alpha=0.2,
                lw=0,
                label=f"{round(100 * ci)}% CI",
            )
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"S = {S}")
            ax.yaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)
            ax.xaxis.grid(color="grey", ls="-", lw=0.5, alpha=0.5)

        axes[0].legend(frameon=True, loc="best")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show_chart:
            plt.show()
        plt.close()

    def plot_premia_term_structure(
            self,
            t=None,
            ci=0.9,
            size=5,
            title=None,
            x_axis_title=r"$S$",
            save_path=None,
            show_chart=True,
    ):
        """
        Plots the term structure of risk premia at conditioning time `t`,
        showing the posterior median across horizons together with a credible
        interval.

        Parameters
        ----------
        t: int, label, or None
            Conditioning time. If a positional integer (zero-indexed in
            `lambda_g_dates`), that row is used. If a label, it is matched
            against `lambda_g_dates`. If None, defaults to the most recent
            available conditioning time.
        """
        if t is None:
            t_idx = len(self.lambda_g_dates) - 1
            label = self.lambda_g_dates[t_idx]
        elif isinstance(t, (int, np.integer)):
            t_idx = int(t)
            label = self.lambda_g_dates[t_idx]
        else:
            t_idx = self.lambda_g_dates.get_loc(t)
            label = t

        if title is None:
            title = rf"$\lambda_{{g,t}}^{{S}}$ at t = {label}"

        snapshot = self.draws_lambda_g[:, t_idx, :]
        median = np.median(snapshot, axis=0)
        lower = np.quantile(snapshot, (1 - ci) / 2, axis=0)
        upper = np.quantile(snapshot, (1 + ci) / 2, axis=0)
        horizons = np.arange(self.s_bar + 1)

        plt.figure(figsize=(size * (16 / 7.3), size))
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.plot(horizons, median, color="tab:blue", label="median")
        ax.fill_between(
            horizons,
            lower,
            upper,
            color="tab:blue",
            alpha=0.2,
            lw=0,
            label=f"{round(100 * ci)}% Credible Interval",
        )
        ax.axhline(0, color="black", lw=0.5)
        ax.set(title=title, xlabel=x_axis_title)
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
        )
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set(title="Medians of Factor Loadings")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show_chart:
            plt.show()
        plt.close()

    def _run_conditional_gibbs(self):
        # ---------------- Setup ----------------
        R = self.assets.values                                       # (T, N)
        Y_dm = R - R.mean(axis=0, keepdims=True)                     # column-demeaned returns
        Z_raw = self.predictors.values                               # (T, p)
        Z_dm = Z_raw - Z_raw.mean(axis=0, keepdims=True)             # column-demeaned predictors
        K = self.k
        p = self.p
        Kp = K + p
        q = self.q
        T = self.t
        T_v = T - q                                                  # number of VAR innovations
        T_g = T_v - self.s_bar                                       # rows of MA design matrix
        G = self.macro_factor.values[self.s_bar + q:].reshape(-1, 1) # (T_g, 1)
        D_r = np.eye(K + 1)
        D_r[0, 0] = 0

        # ---------------- PCA initialization ----------------
        pca = PCA(n_components=K)
        V0 = pca.fit_transform(self.assets)                          # (T, K)
        ups = V0.T                                                   # (K, T)

        # Initial restricted VAR: regress [v_t, z_t] on [1, z_{t-1}, ..., z_{t-q}]
        x_full = np.hstack([ups.T, Z_dm])                            # (T, K+p)
        X1_init = x_full[q:]                                         # (T_v, K+p)
        X0_blocks = [np.ones((T_v, 1))]
        for lag in range(1, q + 1):
            X0_blocks.append(Z_dm[q - lag : T - lag])
        X0_init = np.concatenate(X0_blocks, axis=1)                  # (T_v, 1 + q*p)
        Phi = inv(X0_init.T @ X0_init) @ X0_init.T @ X1_init         # (1 + q*p, K+p)
        eps_v_init = (X1_init - X0_init @ Phi)[:, :K].T              # (K, T_v)

        # Initial MA regression of G on stacked lags of eps_v to seed eta_g, rho_g
        ma_stack = np.empty((T_g, (1 + self.s_bar) * K))
        for ss in range(1 + self.s_bar):
            ma_stack[:, ss * K : (ss + 1) * K] = eps_v_init[:, self.s_bar - ss : self.s_bar - ss + T_g].T
        X_ma = sm.add_constant(ma_stack)
        ma_res = sm.OLS(G.flatten(), X_ma).fit()
        beta_mat = ma_res.params[1:].reshape(1 + self.s_bar, K).T    # (K, 1+s_bar)
        U_svd, s_sv, Vt_svd = svd(beta_mat, full_matrices=False)
        eta_g = U_svd[:, [0]]                                        # (K, 1)
        rho_g = np.concatenate([[ma_res.params[0]], Vt_svd[0, :] * s_sv[0]]).reshape(-1, 1)  # (2+s_bar, 1)
        eps_v = eps_v_init                                           # (K, T_v) — used in next step 1

        # ---------------- Pre-allocate ----------------
        total_draws = self.n_draws + self.burnin
        n_phi_rows = 1 + q * p
        draws_lambda_g = np.empty((total_draws, T, self.s_bar + 1))
        draws_lambda_g_uncond_arr = np.empty((total_draws, self.s_bar + 1))
        draws_loadings_arr = np.empty((total_draws, self.n * K))
        draws_eta_g_arr = np.empty((total_draws, K))
        draws_rho_arr = np.empty((total_draws, self.s_bar + 1))
        draws_Sigma_eps_v_arr = np.empty((total_draws, K * K))
        draws_Sigma_r_arr = np.empty((total_draws, self.n * self.n))
        draws_Phi_arr = np.empty((total_draws, n_phi_rows * Kp))
        draws_Sigma_eps_x_arr = np.empty((total_draws, Kp * Kp))

        for dd in tqdm(range(total_draws)):

            # ----- STEP 1: sample eta_g, then rho_g (R order, HelperFuncs_var1.R:101-143) -----
            V_array = np.empty((T_g, 1 + self.s_bar, K))
            for ss in range(1 + self.s_bar):
                V_array[:, ss, :] = eps_v[:, self.s_bar - ss : self.s_bar - ss + T_g].T

            # (1a) eta_g: project V_array by current rho_g[1:] then add intercept
            Xg_eta = np.einsum('nsk,s->nk', V_array, rho_g[1:, 0])    # (T_g, K)
            Xg_eta = np.column_stack([np.ones(T_g), Xg_eta])          # (T_g, K+1)
            XtX_eta_inv = inv(Xg_eta.T @ Xg_eta)
            eta_full_hat = XtX_eta_inv @ Xg_eta.T @ G                 # (K+1, 1)
            weta_hat = G - Xg_eta @ eta_full_hat
            Sigma_hat_eta = _build_Sigma_hat(Xg_eta, T_v, self.s_bar, weta_hat)
            eta_full_draw = multivariate_normal.rvs(
                mean=eta_full_hat.flatten(),
                cov=nearest_psd(Sigma_hat_eta),
            )
            eta_g = eta_full_draw[1:].reshape(-1, 1)                  # drop intercept
            eta_g = eta_g / np.sqrt(eta_g.T @ eta_g)
            draws_eta_g_arr[dd] = eta_g.flatten()

            # (1b) rho_g: build V_rho via projection by just-sampled eta_g, with intercept
            V_rho = _build_V_rho(eps_v, eta_g, self.s_bar, T_v)       # (T_g, 2+s_bar)
            XtX_rho_inv = inv(V_rho.T @ V_rho)
            rho_hat = XtX_rho_inv @ V_rho.T @ G                       # (2+s_bar, 1)
            wg_hat = G - V_rho @ rho_hat
            Sigma_hat_rho = _build_Sigma_hat(V_rho, T_v, self.s_bar, wg_hat)
            rho_g = multivariate_normal.rvs(
                mean=rho_hat.flatten(),
                cov=nearest_psd(Sigma_hat_rho),
            ).reshape(-1, 1)
            draws_rho_arr[dd] = rho_g[1:, 0]

            # ----- STEP 2: sample (Sigma_wr, B_r) for asset returns (R:151-168) -----
            V_r = np.column_stack([np.ones(T), ups.T])                # (T, K+1)
            A_step2 = V_r.T @ V_r + D_r
            B_mean = np.linalg.solve(A_step2, V_r.T @ R)              # (K+1, N) — Pi.T in R
            resid_step2 = R - V_r @ B_mean                            # (T, N)
            SSR_diag = np.diag(resid_step2.T @ resid_step2)           # (N,)

            sigma_wr_diag = invgamma.rvs(
                0.5 * (T - 1 - K),
                scale=0.5 * SSR_diag,
            )
            Sigma_wr = np.diag(sigma_wr_diag)

            B_r = matrix_normal.rvs(
                mean=B_mean,
                rowcov=nearest_psd(inv(A_step2)),
                colcov=nearest_psd(Sigma_wr),
            )                                                         # (K+1, N)
            beta_ups = B_r.T[:, 1:]                                   # (N, K)
            mu_r_draw = B_r[0, :].reshape(-1, 1)                      # (N, 1) — sampled intercept
            draws_loadings_arr[dd] = beta_ups.flatten()

            # ----- STEP 3: sample latent factors v_t (R:193-196) — adds I_K prior -----
            beta_scaled = beta_ups / sigma_wr_diag[:, None]           # β / σ²_wr_i
            cov_v = nearest_psd(inv(beta_scaled.T @ beta_ups + np.eye(K)))
            v_hat = cov_v @ beta_scaled.T @ Y_dm.T                    # (K, T)
            L = cholesky(cov_v, lower=True)
            ups = v_hat + L @ norm.rvs(size=(K, T))                   # (K, T)

            # ----- STEP 4: sample restricted VAR (R:199-227) -----
            x_full = np.hstack([ups.T, Z_dm])                         # (T, K+p)
            X1 = x_full[q:]                                           # (T_v, K+p)
            X0_blocks = [np.ones((T_v, 1))]
            for lag in range(1, q + 1):
                X0_blocks.append(Z_dm[q - lag : T - lag])
            X0 = np.concatenate(X0_blocks, axis=1)                    # (T_v, 1 + q*p)

            XtX = X0.T @ X0
            XtX_inv = inv(XtX)
            Phi_hat = XtX_inv @ X0.T @ X1                             # (1 + q*p, K+p)
            resid_var = X1 - X0 @ Phi_hat
            resid_dm = resid_var - resid_var.mean(axis=0, keepdims=True)
            Sigma_eps_x_hat = resid_dm.T @ resid_dm / (T_v - 1)
            scale_iw = nearest_psd((T_v - 1) * Sigma_eps_x_hat)

            if Kp == 1:
                Sigma_eps_x = np.array([[invwishart.rvs(df=T_v - p, scale=scale_iw)]])
            else:
                Sigma_eps_x = invwishart.rvs(df=T_v - p, scale=scale_iw)
            Phi = matrix_normal.rvs(
                mean=Phi_hat,
                rowcov=nearest_psd(XtX_inv),
                colcov=nearest_psd(Sigma_eps_x),
            )                                                         # (1 + q*p, K+p)
            Sigma_eps_v = np.atleast_2d(Sigma_eps_x[:K, :K])
            draws_Phi_arr[dd] = Phi.flatten()
            draws_Sigma_eps_x_arr[dd] = Sigma_eps_x.flatten()
            draws_Sigma_eps_v_arr[dd] = Sigma_eps_v.flatten()

            # Demean states by the just-sampled factor intercept (R:223)
            mu_v = Phi[0, :K]
            ups = ups - mu_v.reshape(-1, 1)

            # Refresh eps_v from OLS residuals for next iteration's step 1 (R:204-205)
            eps_v = (X1 - X0 @ Phi_hat)[:, :K].T                      # (K, T_v)

            # ----- STEP 5: lambda_v, unconditional lambda_g, time-varying lambda_g -----
            Sigma_r = beta_ups @ Sigma_eps_v @ beta_ups.T + Sigma_wr  # (N, N)
            mu_tilde = mu_r_draw + 0.5 * np.diag(Sigma_r).reshape(-1, 1)
            lambda_v = inv(beta_ups.T @ beta_ups) @ beta_ups.T @ mu_tilde   # (K, 1)
            lambda_f = (lambda_v.T @ eta_g).item()
            draws_Sigma_r_arr[dd] = Sigma_r.flatten()

            # Per-period unconditional lambda_g[S] = ρ_S * λ_f (R helper line 189).
            lambda_g_per = rho_g[1:, 0] * lambda_f                    # (1+s_bar,)

            # Time-varying lambda_g (R:229-237): generalize R's q=1 phi1_tilde to q>=1 via
            # a companion form on [v_t, z_t, z_{t-1}, ..., z_{t-q+1}].
            state_size = K + q * p
            M_trans = np.zeros((state_size, state_size))
            for lag in range(1, q + 1):
                phi_lag = Phi[1 + (lag - 1) * p : 1 + lag * p, :]     # (p, K+p)
                phi_v_lag = phi_lag[:, :K].T                          # (K, p)
                phi_z_lag = phi_lag[:, K:].T                          # (p, p)
                M_trans[:K, K + (lag - 1) * p : K + lag * p] = phi_v_lag
                M_trans[K : K + p, K + (lag - 1) * p : K + lag * p] = phi_z_lag
            for lag in range(1, q):
                M_trans[K + lag * p : K + (lag + 1) * p,
                        K + (lag - 1) * p : K + lag * p] = np.eye(p)

            # State at t = [v_t, z_t, z_{t-1}, ..., z_{t-q+1}]; for t < q-1 we pad
            # the unavailable lags of z with zero (Z_dm has zero mean).
            Z_dm_padded = np.vstack([np.zeros((q - 1, p)), Z_dm]) if q > 1 else Z_dm
            y_states = np.empty((T, state_size))
            y_states[:, :K] = ups.T
            for lag in range(q):
                y_states[:, K + lag * p : K + (lag + 1) * p] = Z_dm_padded[(q - 1) - lag : (q - 1) - lag + T]

            # v_pred[t, h, :] = (M^h @ y_states[t])[:K]
            v_pred_dot_eta = np.empty((T, self.s_bar + 2))
            Mh = np.eye(state_size)
            v_pred_dot_eta[:, 0] = ups.T @ eta_g.flatten()            # h=0 (unused; placeholder)
            for h in range(1, self.s_bar + 2):
                Mh = Mh @ M_trans
                v_pred_dot_eta[:, h] = (y_states @ Mh.T)[:, :K] @ eta_g.flatten()

            lambda_g_ts_per = np.zeros((T, 1 + self.s_bar))
            rho_arr = rho_g[1:, 0]                                    # ρ_0..ρ_{s_bar}
            for S in range(1 + self.s_bar):
                # sum_{tau=0..S} rho_arr[tau] * v_pred_dot_eta[:, S + 1 - tau]
                for tau in range(S + 1):
                    lambda_g_ts_per[:, S] += rho_arr[tau] * v_pred_dot_eta[:, S + 1 - tau]

            # This converts the per-period MA representation into the cumulative-average term structure that
            # the paper plots in Figures 5 and 6. The plotted time-varying λ_g is the sum of the time-varying and
            # unconditional components per draw.
            horizon_div = np.arange(1, self.s_bar + 2, dtype=float)
            lambda_g_uncond_t = np.cumsum(np.cumsum(lambda_g_per)) / horizon_div
            lambda_g_ts_t = np.cumsum(lambda_g_ts_per, axis=1) / horizon_div[None, :]

            draws_lambda_g_uncond_arr[dd] = lambda_g_uncond_t
            draws_lambda_g[dd] = lambda_g_ts_t + lambda_g_uncond_t[None, :]

        # ---------------- Convert arrays to DataFrames (post-burnin only) ----------------
        post = slice(-self.n_draws, None)
        loadings_columns = [
            f"{a} - loading {v + 1}"
            for a, v in product(self.assets.columns, range(K))
        ]
        eta_g_columns = [f"eta_g_{v + 1}" for v in range(K)]
        rho_columns = [f"rho_{l}" for l in range(self.s_bar + 1)]
        Sigma_eps_v_columns = [
            f"Sigma_eps_v_{i + 1}_{j + 1}"
            for i, j in product(range(K), range(K))
        ]
        Sigma_r_columns = [
            f"Sigma_r_{a}_{b}"
            for a, b in product(self.assets.columns, self.assets.columns)
        ]
        x_labels = [f"v_{i + 1}" for i in range(K)] + list(self.predictors.columns)
        Phi_row_labels = ["const"]
        for lag in range(1, q + 1):
            Phi_row_labels += [f"lag{lag}_{name}" for name in self.predictors.columns]
        Phi_columns = [
            f"Phi[{r},{c}]"
            for r, c in product(Phi_row_labels, x_labels)
        ]
        Sigma_eps_x_columns = [
            f"Sigma_eps_x_{a}_{b}"
            for a, b in product(x_labels, x_labels)
        ]
        lambda_g_uncond_columns = [f"S_{S}" for S in range(self.s_bar + 1)]

        df_loadings = pd.DataFrame(draws_loadings_arr[post], columns=loadings_columns)
        df_eta_g = pd.DataFrame(draws_eta_g_arr[post], columns=eta_g_columns)
        df_rho = pd.DataFrame(draws_rho_arr[post], columns=rho_columns)
        df_Sigma_eps_v = pd.DataFrame(draws_Sigma_eps_v_arr[post], columns=Sigma_eps_v_columns)
        df_Sigma_r = pd.DataFrame(draws_Sigma_r_arr[post], columns=Sigma_r_columns)
        df_Phi = pd.DataFrame(draws_Phi_arr[post], columns=Phi_columns)
        df_Sigma_eps_x = pd.DataFrame(draws_Sigma_eps_x_arr[post], columns=Sigma_eps_x_columns)
        df_lambda_g_uncond = pd.DataFrame(draws_lambda_g_uncond_arr[post], columns=lambda_g_uncond_columns)
        lambda_g_post = draws_lambda_g[post]

        return (
            lambda_g_post,
            df_lambda_g_uncond,
            df_loadings,
            df_eta_g,
            df_rho,
            df_Sigma_eps_v,
            df_Sigma_r,
            df_Phi,
            df_Sigma_eps_x,
        )

    def _get_number_latent_factors(self):
        retp_ret = ((self.assets - self.assets.mean()).T @ (self.assets - self.assets.mean())).values
        eigv = np.sort(eigvals(retp_ret).real)[::-1]
        eigv_normalized = eigv / (self.t * self.n)
        gamma_hat = np.median(eigv_normalized[:self.k_max])
        phi_nt = 0.5 * gamma_hat * np.log(self.t * self.n) * (self.t ** (-0.5) + self.n ** (-0.5))
        j = np.arange(1, self.k_max + 1)
        grid = eigv_normalized[:self.k_max] + j * phi_nt
        k_hat = max(1, np.argmin(grid))
        print("selected number of factors is", k_hat)
        return k_hat

    @staticmethod
    def _assertions(assets, factor, predictors):
        assert factor.index.equals(assets.index), \
            "the index for `factor` and `assets` must match"
        assert predictors is not None, \
            "`predictors` is required: the restricted VAR has latent factors load only on lagged predictors"
        assert predictors.index.equals(assets.index), \
            "the index for `predictors` and `assets` must match"


# ================================================================
# ===== Bayesian Fama-MacBeth - Bryzgalova, Huang & Julliard =====
# ================================================================
class FM:
    # TODO Replace this class with a not-so-simple Fama-MacBeth

    def __init__(self, assets, factors):
        """
        Very simple implementation of the Fama-MacBeth regression using OLS.

        Parameters
        ----------
        assets : pandas.DataFrame
            returns of the test assets

        factors : pandas.DataFrame
            returns of the factor portfolios
        """

        self.beta = pd.DataFrame(
            data=(inv(factors.T @ factors) @ factors.T @ assets).values,
            columns=assets.columns,
            index=factors.columns,
        ).T

        mu = assets.mean()
        self.lambdas = pd.Series(
            data=(inv(self.beta.T @ self.beta) @ self.beta.T @ mu).values,
            index=factors.columns,
            name="Lambdas",
        )


class BFM:
    """
    This class works as both the baseline implementation of the Bayesian
    Fama-MacBeth and the parent class for other variations, which follow the
    same steps, but with different computations.
    """

    def __init__(self, assets, factors, n_draws=1000):
        """
        Implementation of the BFM-OLS

        Parameters
        ----------
        assets : pandas.DataFrame
            returns of the test assets

        factors : pandas.DataFrame
            returns of the factor portfolios

        n_draws : int
            number of draws from the posterior
        """
        self.assets = assets
        self.factors = factors
        self.n_draws = n_draws
        self.y = pd.concat([assets, factors], axis=1).dropna()

        self.t = self.y.shape[0]  # Sample size of the timeseries dimension
        self.n = assets.shape[1]  # Number of test assets
        self.k = factors.shape[1]  # Number of factors

        self.mu_y = self.y.mean()
        self.Sigma_y = self.y.cov()

        self.draws_mu_y, self.draws_Sigma_y = self._draw_mu_sigma()
        self.draws_betas = self._compute_betas()

        self.draws_lambdas = self._compute_lambdas()
        self.draws_r2 = self._compute_r2()

        # Reorganize in Pandas
        self.draws_lambdas = pd.DataFrame(
            data=self.draws_lambdas,
            columns=factors.columns
        )
        self.draws_r2 = pd.Series(
            data=self.draws_r2,
            name="R2",
        )

    def _draw_mu_sigma(self):
        # Draws covariance from inverse wishart
        draws_sigma = invwishart.rvs(
            df=self.t - 1,
            scale=self.t * self.Sigma_y,
            size=self.n_draws,
        )

        # Draws mu from normal, conditional on previous covariances
        draws_mu = np.array(
            [
                multivariate_normal.rvs(
                    mean=self.mu_y,
                    cov=(1 / self.t) * cov,
                )
                for cov in draws_sigma
            ]
        )

        return draws_mu, draws_sigma

    def _compute_betas(self):
        draws_beta = np.array(
            [
                cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:])
                for cov in self.draws_Sigma_y
            ]
        )
        return draws_beta

    def _compute_lambdas(self):
        # campute lambdas from betas and mus
        draws_lambda = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for mu, b in zip(self.draws_mu_y, self.draws_betas)
            ]
        )

        return draws_lambda

    def _compute_r2(self):
        mu_r = self.mu_y[:self.n].values
        denom = (mu_r - mu_r.mean()) @ (mu_r - mu_r.mean())
        draws_r2 = [
            1 - ((mu_r - b @ l) @ (mu_r - b @ l)) / denom
            for b, l in zip(self.draws_betas, self.draws_lambdas)
        ]
        return draws_r2

    def plot_lambda(self, include_fm=False):
        """
        Plots the posterior distributions for the risk premia

        Parameters
        ----------
        include_fm : bool
            If true, adds a vertical line indicating the values for the
            canonical OLS Fama-MacBeth regression.
        """
        axes = self.draws_lambdas.hist(
            density=True,
            bins=int(np.sqrt(self.n_draws)),
            sharex=True,
            figsize=(10, 6)
        )

        if include_fm:
            fm = FM(self.assets, self.factors)
            for ax in axes.flatten():
                try:
                    ax.axvline(
                        fm.lambdas[ax.title.get_text()],
                        color='tab:orange',
                        lw=2,
                    )
                except KeyError:
                    continue

        plt.tight_layout()
        plt.show()

    def plot_r2(self):
        """
        Plots the posteurior distribution for the R2
        """
        axes = self.draws_r2.hist(
            density=True,
            bins=int(np.sqrt(self.n_draws)),
            figsize=(10, 6)
        )
        plt.tight_layout()
        plt.show()

    def ci_table_lambda(self, cred=0.95):
        """
        Return a DataFrame with the median and the credible interval with the
        chosen level of credibility for the risk premia

        Parameters
        ----------
        cred : float
            number between 0 and 1 indicating the credibility level of the
            interval
        """
        table = self.draws_lambdas.quantile(
            q=[
                (1 - cred) / 2,
                0.5,
                (1 + cred) / 2,
            ],
        )
        table.index.name = "Quantiles"
        return table

    def ci_table_r2(self, cred=0.95):
        """
        Return a DataFrame with the median and the credible interval with the
        chosen level of credibility for the R2

        Parameters
        ----------
        cred : float
            number between 0 and 1 indicating the credibility level of the
            interval
        """
        table = self.draws_r2.quantile(
            q=[
                (1 - cred) / 2,
                0.5,
                (1 + cred) / 2,
            ],
        )
        table.index.name = "Quantiles"
        return table


class BFMGLS(BFM):
    """
    This class inherits all the methods and constructor from BFM but modifies
    the computation of the risk premia and R2 to use the GLS precision matrix
    """

    def _compute_lambdas(self):
        # compute idiosyncratic error covariance
        draws_sige = np.array(
            [
                cov[:self.n, :self.n] - cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:]) @ cov[:self.n, -self.k:].T
                for cov in self.draws_Sigma_y
            ]
        )

        # campute lambdas from betas and mus
        draws_lambda = np.array(
            [
                inv(b.T @ inv(sige) @ b) @ b.T @ inv(sige) @ mu[:self.n]
                for mu, b, sige in zip(self.draws_mu_y, self.draws_betas, draws_sige)
            ]
        )

        return draws_lambda

    def _compute_r2(self):
        draws_sige = np.array(
            [
                cov[:self.n, :self.n] - cov[:self.n, -self.k:] @ inv(cov[-self.k:, -self.k:]) @ cov[:self.n, -self.k:].T
                for cov in self.draws_Sigma_y
            ]
        )

        mu_r = self.mu_y[:self.n].values
        mu_r_bar = mu_r.mean()

        draws_r2 = [
            1 - ((mu_r - b @ l) @ inv(sige) @ (mu_r - b @ l)) / ((mu_r - mu_r_bar) @ inv(sige) @ (mu_r - mu_r_bar))
            for b, l, sige in zip(self.draws_betas, self.draws_lambdas, draws_sige)
        ]
        return draws_r2


class BFMOMIT(BFM):
    """
    This class inherits all the methods and constructor from BFM but modifies
    the computation of the risk premia and R2 to use the BFM-OMIT methodology
    """

    def __init__(self, assets, factors, n_draws=1000, p=5):
        """
        Implementation of the BFM-OMIT

        Parameters
        ----------
        assets : pandas.DataFrame
            returns of the test assets

        factors : pandas.DataFrame
            returns of the factor portfolios

        n_draws : int
            number of draws from the posterior

        p : int
            number of principal components to use
        """
        self.p = p
        super().__init__(assets, factors, n_draws)

    def _compute_lambdas(self):

        def cov_svd(cov):
            u, s, _ = svd(cov)
            return (u @ np.diag(s))[:, :self.p]

        draws_beta_upsilon = np.array(
            [
                cov_svd(cov[:self.n, :self.n])
                for cov in self.draws_Sigma_y
            ]
        )

        draws_lambda_upsilon = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for b, mu in zip(draws_beta_upsilon, self.draws_mu_y)
            ]
        )

        draws_lambda_f = np.array(
            [
                lu.T @ inv(bu.T @ bu) @ bu.T @ cov[:self.n, -self.k:]
                for lu, bu, cov in zip(draws_lambda_upsilon, draws_beta_upsilon, self.draws_Sigma_y)
            ]
        )

        return draws_lambda_f

    def _compute_r2(self):

        def cov_svd(cov):
            u, s, _ = svd(cov)
            return (u @ np.diag(s))[:, :self.p]

        draws_beta_upsilon = np.array(
            [
                cov_svd(cov[:self.n, :self.n])
                for cov in self.draws_Sigma_y
            ]
        )

        draws_lambda_upsilon = np.array(
            [
                inv(b.T @ b) @ b.T @ mu[:self.n]
                for b, mu in zip(draws_beta_upsilon, self.draws_mu_y)
            ]
        )

        mu_r = self.mu_y[:self.n].values
        mu_r_bar = mu_r.mean()
        draws_r2 = np.array(
            [
                1 - ((mu_r - bu @ lu) @ (mu_r - bu @ lu)) / ((mu_r - mu_r_bar) @ (mu_r - mu_r_bar))
                for lu, bu in zip(draws_lambda_upsilon, draws_beta_upsilon)
            ]
        )
        return draws_r2
