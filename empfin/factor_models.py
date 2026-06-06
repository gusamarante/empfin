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





# =============================================================
# ===== Macro Strikes Back - Bryzgalova, Huang & Julliard =====
# =============================================================



# ================================================================
# ===== Bayesian Fama-MacBeth - Bryzgalova, Huang & Julliard =====
# ================================================================
