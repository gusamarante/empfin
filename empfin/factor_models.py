import pandas as pd
import numpy as np
from numpy.linalg import inv


class TimeseriesReg:

    def __init__(self, assets, factors):
        # TODO Documentation (factors must be excess returns)
        # TODO ensure same index on assets and factors, must be pandas df
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
        self.params = pd.DataFrame(
            data=Bhat,
            index=factors.columns,
            columns=assets.columns,
        )

    def grs_test(self):
        # TODO documentation
        # TODO parei aqui, eq 12.6 do Cochrane
        pass


class TwoPassOLS:
    pass

class FamaMacbeth:
    pass

class GMM:  # TODO better name
    pass