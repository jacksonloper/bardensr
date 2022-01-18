import pandas as pd
import numpy as np

def locs_and_j_to_df(locs, j):
    '''
    Take a set of positions and colors, and
    turn it into a dataframe which represents
    a colored pointcloud.

    Input:

    - locs (N x 3 float array)
    - j (N integer array)

    Output is a dataframe with columns m0,m1,m2,j
    '''
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
    '''
    Take a dataframe representing a pointcloud
    and turn it into an array of locations
    and an array of colors.

    Input is a dataframe with columns m0,m1,m2,j

    Output:

    - locs (N x 3 float array)
    - j (N integer array)
    '''
    return np.array(df[['m0', 'm1', 'm2']]), np.array(df['j'])
