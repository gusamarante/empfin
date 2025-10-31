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
        self.risk_premia = factors.mean()
        self.Omega = factors.cov()

        # OLS
        factors.insert(0, "alpha", 1)
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
        f2 = 1 / (1 + self.risk_premia.T @ inv(self.Omega) @ self.risk_premia)
        alpha = self.params.loc['alpha']
        f3 = alpha.T @ inv(self.Sigma) @ alpha
        grs = f1 * f2 * f3
        pvalue = 1 - f.cdf(grs, dfn=self.N, dfd=self.T - self.N - self.K)
        return grs, pvalue


class TwoPassReg:

    def __init__(self, assets, factors, cs_const=False):
        # TODO DOcumentation

        ts_reg = TimeseriesReg(assets.copy(), factors.copy())
        self.betas = ts_reg.params.drop('alpha').T
        self.avg_ret = assets.mean()

        if cs_const:
            self.betas.insert(0, "const", 1)

        X = self.betas.values
        Lhat = inv(X.T @ X) @ X.T @ self.avg_ret
        self.lambdas = pd.Series(
            data=Lhat,
            index=["const"] + list(factors.columns) if cs_const else factors.columns,
        )
        self.alphas = pd.Series(
            data=self.avg_ret - X @ Lhat,
            index=assets.columns,
        )
        # TODO cov alpha e lambda. Eq 12.16 do Cochrane

class FamaMacbeth:
    pass

class GMM:  # TODO better name
    pass