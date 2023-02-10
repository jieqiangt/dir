
import pandas as pd
import numpy as np

from utils.de import read_in_df, correct_col_names
from de_func import adi_cv_calc, calc_safety_stock, backcalculate_inv
from de_func import filter_materials, analyse_materials, convert_gr_date
from de_func import get_latest_gr_date, get_gr_qty, calc_leadtime_fl, consolidate_df


def hist_lead_time_calc():

    # Read in data
    main_df = read_in_df('../data/PR.xlsx')

    # data cleaning
    main_df = correct_col_names(main_df)
    # remove purchase docs that have no goods received..
    # either requested but not ordered or change order number
    main_df.dropna(subset=['gr_info'], inplace=True)
    main_df.rename(columns={'po_info': 'po_info_0',
                            'gr_info': 'gr_info_0'}, inplace=True)

    # converting goods received text date into actual datetime & received quantity
    for col_no in range(0, 6):

        temp_df = main_df[f'gr_info_{col_no}'].str.split(" ", expand=True)
        gr_df = convert_gr_date(temp_df, col_no)

        for idx, row in temp_df.iterrows():

            gr_qty = get_gr_qty(row)
            gr_df.loc[idx, f'gr_qty_{col_no}'] = gr_qty

        main_df[f'gr_date_{col_no}'] = gr_df[f'gr_date_{col_no}']
        main_df[f'gr_qty_{col_no}'] = gr_df[f'gr_qty_{col_no}']

    # get the latest date where all goods are received
    main_df = get_latest_gr_date(main_df)

    main_df['req_date'] = pd.to_datetime(main_df['req_date'])
    main_df['lt_days'] = main_df['gr_date'] - main_df['req_date']

    req_cols = ['req_date', 'material', 'lt_days']
    output_df = main_df[req_cols]
    # converting from timedelta to floating point
    output_df['lt_days'] = output_df['lt_days'].dt.days

    return output_df


def latest_lead_time_calc(df):

    # get rolling median with required window
    # min_periods = 1 to prevent null values
    no_of_past_pr = 10
    df['median_lt_days'] = df.groupby('material')['lt_days'].transform(
        lambda s: s.rolling(no_of_past_pr, min_periods=1).median())

    # getting index of max date for each material
    # note that there are multiple PRs during the max date thus there are multiple rows of max date for some material.
    max_date_idx = df.groupby(['material'])[
        'req_date'].transform(max) == df['req_date']
    output_df = df[max_date_idx]

    # taking the largest median lead time for worse case scenario
    output_df = output_df.groupby('material')[['median_lt_days']].max()
    output_df['lt_cv_days'] = (df.groupby(
        'material')['lt_days'].std() / df.groupby('material')['lt_days'].mean())**2
    # dropping material codes with only one data point thus without standard deviation
    output_df.dropna(inplace=True)
    output_df.reset_index(inplace=True)

    return output_df


def hist_demand_purchase_calc():

    main_df = read_in_df('./data/all/mb51.xlsx',
                         skiprows=3, usecols=range(1, 23))
    main_df = correct_col_names(main_df)

    req_cols = ['Pstng Date', 'Material', 'Quantity']
    main_df = main_df.loc[:, req_cols]

    rename_dict = {'Pstng Date': 'date',
                   'Material': 'material', 'Quantity': 'qty'}
    main_df.rename(columns=rename_dict, inplace=True)
    main_df["material"] = main_df["material"].astype("str")

    demand_df = main_df.loc[main_df['qty'] < 0, :]
    purchase_df = main_df.loc[main_df['qty'] > 0, :]

    purchase_materials = purchase_df['material'].unique()
    demand_materials = demand_df['material'].unique()

    unique_materials = analyse_materials(
        main_df, purchase_materials, demand_materials)

    output_df = filter_materials(main_df, unique_materials)

    return output_df


def demand_profile_cat(df):

    cat_dict = {}

    for material in df['material'].unique():

        temp_df = df[df['material'] == material]
        cat_tuple = adi_cv_calc(temp_df)
        cat_dict[material] = cat_tuple

    cat_df = pd.DataFrame(cat_dict).T
    cat_df.columns = ['adi_days', 'cv_days', 'cat_days', 'adi_weeks', 'cv_weeks',
                      'cat_weeks', 'adi_months', 'cv_months', 'cat_months', 'last_demand_date']
    cat_df.reset_index(inplace=True)
    cat_df.rename(columns={'index': 'material'}, inplace=True)
    cat_df = calc_leadtime_fl(cat_df)

    return cat_df


def soh_calc(materials):

    # read in raw data
    main_df = read_in_df('./data/soh.xlsx', skiprows=1, usecols=range(2, 23))

    # Getting and renaming required columns
    main_df = correct_col_names(main_df)
    req_cols = ['Material', 'Unrestricted']
    main_df = main_df[req_cols]
    main_df.rename(columns={'Material': 'material',
                            'Unrestricted': 'inv'}, inplace=True)
    main_df["material"] = main_df["material"].astype("str")

    # assuming adding up all rows is the current inv
    main_df = main_df[main_df['material'].isin(materials)]
    output_df = main_df.groupby('material').sum().reset_index()

    return output_df


def inv_backcalc(df, inv_df):

    base_df = df.reset_index(names='date')

    time_series_df_store = []
    inv_materials = inv_df['material'].unique()

    for material in inv_materials:

        temp_inv_df = backcalculate_inv(base_df, inv_df, material)
        time_series_df_store.append(temp_inv_df)

    output_df = pd.concat(time_series_df_store)

    output_df.reset_index(drop=True, inplace=True)
    output_df.set_index("date", inplace=True)

    return output_df


def dues_out_calc():

    df = read_in_df('./data/dues_out.xlsx', skiprows=4, usecols=range(1, 24))
    df = correct_col_names(df)

    do_req_cols = ['stock_code', 'qty_do', 'do_date',
                   'po_edd', 'qty_di', 'mid_bal', 'blk_stk', 'st_bal']
    output_df = df[do_req_cols]
    output_df.rename(columns={'stock_code': 'material'}, inplace=True)
    output_df['do_date'] = pd.to_datetime(df['do_date'], format='%d.%m.%Y')
    output_df['po_edd'] = pd.to_datetime(df['po_edd'], format='%d.%m.%Y')

    return output_df


def safety_stock_calc(inv_df, cat_df):

    df_collect = []

    for material in inv_df['material'].unique():

        temp_df = calc_safety_stock(inv_df, cat_df, material)
        df_collect.append(temp_df)

    output_df = pd.concat(df_collect)
    output_df.reset_index(inplace=True)

    return output_df
