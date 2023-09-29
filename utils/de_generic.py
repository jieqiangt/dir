import pandas as pd
import numpy as np
from datetime import datetime

def remove_unnamed_cols(df):
    '''
        Removes all columns that are unnamed when read in using pd.read_excel.

        Parameters
        ----------
        df: Pandas DataFrame
            DataFrame read in using function pd.read_excel with appropriate skiprows & usecols argument
        Returns
        -------
        df : Pandas DataFrame
            Dataframe without unnamed columns.
    '''

    for col in filter(lambda col: col.find('Unnamed') != -1, df.columns):
        df.drop(columns=col, inplace=True)

    return df


def correct_col_names(df):

    new_cols = []

    for col in df.columns:

        col = col.strip()
        col = col.replace(".", "_")
        col = col.replace(" ", "_")
        col = col.replace("-", "_")
        col = col.lower()
        new_cols.append(col)

    df.columns = new_cols

    return df


def read_in_df(df_path, skiprows=None, usecols=None):
    '''
        Creates Pandas DataFrame by reading excel file from provided path.

        Parameters
        ----------
        df_path: str
            Path to excel file to be read in.
        skiprows: int
            Number of rows at the start of excel to be skipped before reading in.
        usecols: List/List-like object
            The index of columns that are to be read in.
        Returns
        -------
        df : Pandas DataFrame
            Dataframe without unnammed columns read in from excel file in specified path.
    '''

    df = pd.read_excel(df_path, skiprows=skiprows, usecols=usecols, header=0)
    df = remove_unnamed_cols(df)
    df = df.dropna(how='all')

    return df


def resample_df(df, rule, start_date, end_date):

    dt_index = pd.date_range(start=start_date, end=end_date, freq=rule)
    df.set_index('date', inplace=True)
    df = df.resample(rule=rule).sum()
    df = df.reindex(dt_index, fill_value=0)
    df.index = pd.to_datetime(df.index)

    return df


def format_time_series(df, material, backtest):

    temp_df = df[df['material'] == material][['date', 'qty']]

    temp_df['date'] = pd.to_datetime(temp_df['date'], format='%d.%m.%Y')
    date_first = temp_df['date'].min()

    if backtest:
        date_last = temp_df['date'].max()
    else:
        # Need to decide on the end date
        date_last = datetime(2021, 4, 18)

    d_df = temp_df.loc[df['qty'] < 0, :].copy()
    p_df = temp_df.loc[df['qty'] > 0, :].copy()

    # change to positive for demand qty
    d_df['qty'] = np.abs(d_df['qty'])

    # resample to day
    d_df = resample_df(
        d_df, rule='D', start_date=date_first, end_date=date_last)
    p_df = resample_df(
        p_df, rule='D', start_date=date_first, end_date=date_last)

    # rename columns to respective columns
    d_df.rename(columns={'qty': 'demand'}, inplace=True)
    p_df.rename(columns={'qty': 'purchases'}, inplace=True)

    final_df = d_df.join(p_df)

    # fill na for no purchases
    final_df.fillna(0, inplace=True)

    # calculate cumulative sum columns
    final_df['cum_demand'] = np.cumsum(final_df['demand'])
    final_df['cum_purchases'] = np.cumsum(final_df['purchases'])

    return final_df


def consolidate_df(base_df, **kwargs):

    materials = base_df['material'].unique()

    for arg in kwargs.values():

        base_df = base_df.merge(arg['material'].isin(materials))

    return base_df
