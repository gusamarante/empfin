import pandas as pd
from pathlib import Path

GITHUB_DATA = Path("https://raw.githubusercontent.com/gusamarante/empfin/refs/heads/main/sample-data")

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
            GITHUB_DATA.joinpath("F-F_Research_Data_5_Factors_2x3.csv"),
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
            GITHUB_DATA.joinpath("25_Portfolios_5x5.csv"),
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

def bond_futures():
    """
    Loads the bond futures excess return indexes from 24 developed countries
    """
    try:  # If repo is cloned, try to read locally for performance
        bonds = pd.read_csv(
            "../sample-data/Bond Futures.csv",
            index_col="date",
            sep=";",
        )
    except FileNotFoundError:  # If fails, when the package is installed, grab online
        bonds = pd.read_csv(
            GITHUB_DATA.joinpath("Bond Futures.csv"),
            index_col="date",
            sep=";",
        )
    bonds.index = pd.to_datetime(bonds.index)
    return bonds


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
            GITHUB_DATA.joinpath("UST Futures.csv"),
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
            GITHUB_DATA.joinpath("CPI.csv"),
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
            GITHUB_DATA.joinpath("GDP.csv"),
            index_col="observation_date",
        )
    gdp.index = pd.to_datetime(gdp.index)
    gdp = gdp.resample("QE").last()["GDP"]
    return gdp
