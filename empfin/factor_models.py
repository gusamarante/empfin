import numpy as np
import pandas as pd
from numpy.linalg import inv, eigvals
from scipy.stats import f


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


class TwoPassGLS:
    # TODO Implement
    pass


class FamaMacbeth:
    # TODO Implement
    pass


class GMM:
    # TODO Implement
    pass


class MacroRiskPremium:
    # TODO Documentation
    k_max = 15

    def __init__(self, assets, macro_factor, s_bar, n_draws=100, k=None):
        # TODO Documentation
        #  None or int. if s_bar is None, select the order automatically

        # Simple attributes
        self.assets = assets
        self.macro_factor = macro_factor
        self.t, self.n = assets.shape
        self.s_bar = s_bar  # TODO can be inferred from time freq
        self.n_draws = n_draws

        # select number of latent factors
        if k is None:
            self.k = self._get_number_latent_factors()
        else:
            assert isinstance(k, int), "`k` must be an integer"
            self.k = k

        # Run the gibbs sampler
        self.draws = self._run_gibbs()

    def _run_gibbs(self):

        # Auxiliar matrices that do NOT update every draw
        G = self.macro_factor.values[self.s_bar:]
        mu_g = self.macro_factor.mean()
        G_bar = G - mu_g

        # Starting points
        ups = np.zeros((self.k, self.t))  # latent factors
        mu_ups = ups.mean(axis=1).reshape(self.k, -1)
        eta_g = np.zeros((self.k, 1))
        rho_g = np.insert(np.zeros(self.s_bar + 1), 0, mu_g).reshape(-1,1)
        V_rho = self._build_V_rho(ups, mu_ups, eta_g)  # TODO parei aqui

        return 1

    @staticmethod
    def _build_V_rho(ups, mu_ups, eta_g):
        return 1  # TODO PAREI AQUI, CONTRUIR ESSA MATRIZ

    def _get_number_latent_factors(self):
        retp_ret = ((self.assets - self.assets.mean()).T @ (self.assets - self.assets.mean())).values
        eigv = np.sort(eigvals(retp_ret))[::-1]
        gamma_hat = np.median(eigv[:self.k_max])
        phi_nt = 0.5 * gamma_hat * np.log(self.t * self.n) * (self.t**(-0.5) + self.n**(-0.5))
        j = (np.arange(len(eigv)) + 1)
        grid = (eigv / (self.t * self.n)) + j * phi_nt
        k_hat  = np.argmin(grid) + 1
        return k_hat

