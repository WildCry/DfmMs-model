import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame, dropna: bool = True, contain_zeros: bool = False) -> pd.DataFrame:
    '''
    Will log difference data and normalize it such that it has zero mean and unit variance, without using sklearn for improved performance.

    Parameters:
    - df: A pandas DataFrame with numeric data.
    - dropna: bool. Wether you want to drop nans from dataframe
    - contain_zeros: bool. Wether DataFrame contains zeros. setting this to True will replace 0 with 1e-9.

    Returns:
    - A pandas DataFrame after applying log difference and normalization.
    '''

    # Adding a small constant to avoid log(0), adjust as needed based on your data
    if contain_zeros:
        df = df.replace(0, 1e-9)

    # Apply log transformation directly without replacing zeros (assumes no zero values)
    df = df.apply(lambda x: np.log(x))

    # Compute the difference more efficiently
    df = df.diff()

    if dropna:
        df = df.dropna()

    # Normalize the data to have zero mean and unit variance using pandas and numpy
    mean = df.mean()
    std = df.std()
    normalized_df = (df - mean) / std

    return normalized_df
