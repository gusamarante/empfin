import pandas as pd
from urllib.parse import urljoin, quote

GITHUB_DATA = "https://raw.githubusercontent.com/gusamarante/empfin/refs/heads/main/sample-data/"

def bond_futures():
    """
    Loads the excess return index for bond futures of 24 developed countries
    """
    try:  # If repo is cloned, try to read locally for performance
        bonds = pd.read_csv(
            "../sample-data/Bond Futures.csv",
            index_col="date",
            sep=";",
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        bonds = pd.read_csv(
            urljoin(GITHUB_DATA, quote("Bond Futures.csv")),
            index_col="date",
            sep=";",
        )
    bonds.index = pd.to_datetime(bonds.index)
    return bonds

def ff5f():
    """
    Loads the Fama-French 5 factors and the risk-free rate
    """
    try:  # If repo is cloned, try to read locally for performance
        factors = pd.read_csv(
            "../sample-data/F-F_Research_Data_5_Factors_2x3.csv",
            index_col="Date",
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        factors = pd.read_csv(
            urljoin(GITHUB_DATA, quote("F-F_Research_Data_5_Factors_2x3.csv")),
            index_col="Date",
        )

    factors.index = pd.to_datetime(factors.index, format="%Y%m")
    factors = factors.resample("ME").last()
    rf = factors.pop("RF")
    return factors, rf

def ff25p(sub_rf=True):
    """
    Loads the Fama-French 25 size-value double-sorted portfolios

    Parameters
    ----------
    sub_rf: bool
        If True, substracts the risk-free rate and returns excess returns
    """
    try:  # If repo is cloned, try to read locally for performance
        ports = pd.read_csv(
            "../sample-data/25_Portfolios_5x5.csv",
            index_col="Date",
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        ports = pd.read_csv(
            urljoin(GITHUB_DATA, quote("25_Portfolios_5x5.csv")),
            index_col="Date",
        )
    ports = ports.rename(
        {
            "SMALL LoBM": "ME1 BM1",
            "SMALL HiBM": "ME1 BM5",
            "BIG LoBM": "ME5 BM1",
            "BIG HiBM": "ME5 BM5",
        },
        axis=1,
    )
    ports.index = pd.to_datetime(ports.index, format="%Y%m")
    ports = ports.resample("ME").last()

    if sub_rf:
        _, rf = ff5f()
        ports = ports.sub(rf, axis=0).dropna(how="all")

    return ports

def msb_replication():
    """
    Loads the data to replicate a result from "Macro Strikes Back: Term Structure of Risk Premia"
    """
    try:  # If repo is cloned, try to read locally for performance
        data_df = pd.read_csv("../sample-data/GDP_data_clean.csv", index_col=0)
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        data_df = pd.read_csv(
            urljoin(GITHUB_DATA, quote("GDP_data_clean.csv")),
            index_col=0,
        )

    data_df.index = pd.to_datetime(data_df.index)
    return data_df

def msb_conditional_replication():
    """
    Loads the authors' replication data for the *conditional* model in
    "Macro Strikes Back: Term Structure of Risk Premia". Both files use a
    sequential integer index in their on-disk form; this function attaches the
    quarterly datetime index 1963Q3..2019Q4 used by the paper.

    Returns
    -------
    data_df: pandas.DataFrame
        FF275 portfolio returns plus a column ``"GDP"`` (standardized real GDP
        growth) on a quarterly datetime index.

    predictors: pandas.DataFrame
        The four external predictors used as ``z_t`` in the VAR
        (``pe_ratio``, ``term_spread``, ``default_spread``, ``value_spread``).
    """
    try:  # If repo is cloned, try to read locally for performance
        data_df = pd.read_csv(
            "../sample-data/conditional/GDP_data.csv",
            index_col=0,
        )
        predictors = pd.read_csv(
            "../sample-data/conditional/external_predictors_data.csv",
            index_col=0,
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        data_df = pd.read_csv(
            urljoin(GITHUB_DATA, quote("conditional/GDP_data.csv")),
            index_col=0,
        )
        predictors = pd.read_csv(
            urljoin(GITHUB_DATA, quote("conditional/external_predictors_data.csv")),
            index_col=0,
        )

    # First column has the R-mangled name for normalized real GDP growth
    data_df = data_df.rename(columns={data_df.columns[0]: "GDP"})
    data_df.columns = data_df.columns.str.replace("as.matrix(window(assets, start = c(1963, 3), end = c(2019, 4))).", "")

    dates = pd.date_range(start="1963-09-30", periods=len(data_df), freq="QE")
    data_df.index = dates
    predictors.index = dates

    return data_df, predictors

def ust_futures():
    """
    Loads the US bond futures excess return indexes for 6 different maturities
    """
    try:  # If repo is cloned, try to read locally for performance
        ust = pd.read_csv(
            "../sample-data/UST Futures.csv",
            index_col="date",
            sep=";",
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        ust = pd.read_csv(
            urljoin(GITHUB_DATA, quote("UST Futures.csv")),
            index_col="date",
            sep=";",
        )
    ust.index = pd.to_datetime(ust.index)
    return ust

def us_cpi():
    """
    Loads the timeseries for the US Headline CPI
    """
    try:  # If repo is cloned, try to read locally for performance
        cpi = pd.read_csv(
            "../sample-data/CPI.csv",
            index_col="observation_date",
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        cpi = pd.read_csv(
            urljoin(GITHUB_DATA, quote("CPI.csv")),
            index_col="observation_date",
        )
    cpi.index = pd.to_datetime(cpi.index)
    cpi = cpi.resample("ME").last()["CPI"]
    return cpi

def us_gdp():
    """
    Loads the timeseries for the US GDP
    """
    try:  # If repo is cloned, try to read locally for performance
        gdp = pd.read_csv(
            "../sample-data/GDP.csv",
            index_col="observation_date",
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        gdp = pd.read_csv(
            urljoin(GITHUB_DATA, quote("GDP.csv")),
            index_col="observation_date",
        )
    gdp.index = pd.to_datetime(gdp.index)
    gdp = gdp.resample("QE").last()["GDP"]
    return gdp

def vix():
    try:  # If repo is cloned, try to read locally for performance
        v = pd.read_excel(
            "../sample-data/VIXCLS.xlsx",
            index_col="observation_date",
        )
    except FileNotFoundError:
        v = pd.read_excel(
            urljoin(GITHUB_DATA, quote("VIXCLS.xlsx")),
            index_col="observation_date",
        )
    v.index = pd.to_datetime(v.index)
    v = v["VIXCLS"].rename("VIX")
    return v
