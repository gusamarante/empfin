import numpy as np
import pandas as pd
from numpy.linalg import inv
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
