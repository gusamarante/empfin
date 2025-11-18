"""
Clean the replication data
"""

import pandas as pd

data_df = pd.read_csv("../sample-data/GDP_data.csv", index_col=0)
dates = pd.date_range(start="1963-09-30", end="2019-12-31", freq="QE")
data_df = data_df.set_index(dates)
data_df = data_df.rename(
    {
        "as.matrix(window(real_gdp_g/sd(real_gdp_g), start = c(1963, 3), ": "GDP",
    },
    axis=1,
)
data_df.columns = data_df.columns.str.replace("as.matrix(window(assets, start = c(1963, 3), end = c(2019, 4))).ff25_", "")
data_df.columns = data_df.columns.str.replace("_q", "")
data_df.columns = data_df.columns.str.replace("[, 1:25]", "")

data_df.to_csv("../sample-data/GDP_data_clean.csv")
