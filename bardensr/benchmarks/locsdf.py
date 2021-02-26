import pandas as pd
import numpy as np

def locs_and_j_to_df(locs, j):
    return pd.DataFrame(dict(
        m0=locs[:, 0],
        m1=locs[:, 1],
        m2=locs[:, 2],
        j=j
    ))


def locsj_to_df(locsj):
    return pd.DataFrame(dict(
        m0=locsj[:, 0],
        m1=locsj[:, 1],
        m2=locsj[:, 2],
        j=locsj[:, 3],
    ))


def df_to_locs_and_j(df):
    return np.array(df[['m0', 'm1', 'm2']]), np.array(df['j'])
