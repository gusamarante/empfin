import pandas as pd


def ff5f():
    # TODO Documentation
    factors = pd.read_csv(
        "../sample-data/F-F_Research_Data_5_Factors_2x3.csv",  # TODO add link to file online
        index_col="Date",
    )
    factors.index = pd.to_datetime(factors.index, format="%Y%m")
    factors = factors.resample("ME").last()
    rf = factors.pop("RF")
    return factors, rf


def ff25p(sub_rf=True):
    # TODO Documentation
    ports = pd.read_csv(
        "../sample-data/25_Portfolios_5x5.csv",  # TODO add link to file online
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
