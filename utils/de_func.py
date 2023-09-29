from utils.de_generic import format_time_series
import numpy as np
import pandas as pd


def calculate_avg_demand(df, lt):

    df['avg_demand'] = df['demand'].transform(
        lambda s: s.rolling(int(lt), min_periods=1).mean())

    return df.loc[:, ['demand', 'avg_demand']]


def calc_safety_stock(temp_df, cat_df, material):

    # get constants required for formula for each mat code
    lt_fl = cat_df.loc[cat_df['material'] == material]['lt_fl'].item()
    d_cv = cat_df.loc[cat_df['material'] == material]['cv_days'].item()
    lt_cv = cat_df.loc[cat_df['material'] == material]['lt_cv_days'].item()
    lt = cat_df.loc[cat_df['material'] == material]['median_lt_days'].item()

    # Yellow Zone (YZ): Average daily demand up till date (D) * average lead time (L)
    # Red Zone (RZ): RB + RS = (D * (1.02 * sqrt(L) + 1.15) * (1 + sqrt(cv_d + cv_l * L))
    # Green Zone (GZ): (D * L) · Fl
    temp_df = temp_df[temp_df['material'] == material]
    temp_df = calculate_avg_demand(temp_df, lt)
    temp_df['YZ'] = temp_df['avg_demand'] * lt
    temp_df['RZ'] = (temp_df['avg_demand'] * (1.02 *
                     np.sqrt(lt) + 1.15)) * (1 + np.sqrt(d_cv + lt_cv * lt))
    temp_df['GZ'] = (temp_df['avg_demand'] * lt) * lt_fl
    temp_df['TOY'] = temp_df['RZ'] + temp_df['YZ']
    temp_df['TOG'] = temp_df['TOY'] + temp_df['GZ']
    temp_df['material'] = material

    return temp_df


def calculate_previous_day_inv(row):

    global temp_running_inv
    cur_inv = temp_running_inv + row['demand'] - row['purchases']
    temp_running_inv = cur_inv

    return cur_inv


def backcalculate_inv(df, inv_df, material):

    global temp_running_inv

    temp_df = df[df['material'] == material]
    temp_inv = inv_df[inv_df['material'] == material]

    temp_df = temp_df.iloc[::-1].reset_index(drop=True)
    temp_start_inv = temp_inv['inv'].item()
    temp_running_inv = temp_start_inv

    temp_df['running_inv'] = temp_df.apply(calculate_previous_day_inv, axis=1)
    temp_df['running_inv'] = temp_df['running_inv'].shift(1)
    temp_df['running_inv'] = temp_df['running_inv'].fillna(temp_start_inv)

    return temp_df


def filter_materials(df, unique_materials):

    # Filtering out materials that have insufficient points of demand
    # will only use material codes with both demand and purchase
    # will only use material codes with at least 20 rows of data

    min_rows = 10
    failed_materials = []
    materials = []
    output_df_store = []

    for material in unique_materials:

        plot_df = format_time_series(df, material, True)
        temp_df = plot_df[plot_df['demand'] > 0]

        if temp_df.shape[0] >= min_rows:
            materials.append(material)
            plot_df['material'] = material
            output_df_store.append(plot_df)
        else:
            failed_materials.append(material)

    print(f'Total unique codes: {len(unique_materials)}')
    print(f'Total leftover codes: {len(materials)}')

    output_df = pd.concat(output_df_store)

    return output_df


def analyse_materials(df, purchase_materials, demand_materials):

    # materials that have both demand & purchase
    unique_materials = np.intersect1d(purchase_materials, demand_materials)

    # materials that appear in demand but not in purchase
    demand_only = np.setdiff1d(demand_materials, purchase_materials)

    # materials that appear in purchase but not in demand
    purchase_only = np.setdiff1d(purchase_materials, demand_materials)

    print(f"All unique materials: {df['material'].nunique()}")
    print(f"materials with demand & purchase data: {len(unique_materials)}")
    print(f"materials with demand only data: {len(demand_only)}")
    print(f"materials with purchase only data: {len(purchase_only)}")

    return unique_materials, demand_only, purchase_only


def get_gr_qty(row):

    start_col = 6

    while row[start_col] in ['', '/']:
        start_col += 1

    gr_qty = row[start_col]

    return gr_qty


def convert_po_date(df):
    # converting purchase order text date into actual datetime

    for col_no in range(0, 3):

        date_col = f'po_info_{col_no}'
        output_df = df[date_col].str.split(" ", expand=True)[[3]]
        output_df.rename(columns={3: f'po_date_{col_no}'}, inplace=True)
        df[f'po_date_{col_no}'] = pd.to_datetime(
            output_df[f'po_date_{col_no}'], format='%Y%m%d')

    return output_df


def convert_gr_date(df, col_no):

    output_df = df[[4]]
    output_df.rename(columns={4: f'gr_date_{col_no}'}, inplace=True)
    output_df[f'gr_date_{col_no}'] = pd.to_datetime(
        output_df[f'gr_date_{col_no}'], format='%Y%m%d')

    return output_df


def get_latest_gr_date(df):

    # get the latest date where all goods are received
    df['gr_date'] = np.max(
        df[[f'gr_date_{col_no}' for col_no in range(0, 6)]], axis=1)

    return df


def adi_cv_calc(df):

    temp_df = df[['demand']]
    temp_df.index.name = 'date'

    # for adi days
    days_df = temp_df.resample(rule='D').sum()
    days_df = days_df[days_df['demand'] > 0].copy()
    adi_days = np.mean(days_df.reset_index()[
                       'date'].diff())/np.timedelta64(1, 'D')
    cv_days = (np.std(days_df['demand'])/np.mean(days_df['demand']))**2
    cat_days = demand_cat(adi_days, cv_days)

    # for adi weeks
    weeks_df = temp_df.resample(rule='W').sum()
    weeks_df = weeks_df[weeks_df['demand'] > 0].copy()
    adi_weeks = np.mean(weeks_df.reset_index()[
                        'date'].diff())/np.timedelta64(1, 'W')
    cv_weeks = (np.std(weeks_df['demand'])/np.mean(weeks_df['demand']))**2
    cat_weeks = demand_cat(adi_weeks, cv_weeks)

    # for adi months
    months_df = temp_df.resample(rule='MS').sum()
    months_df = months_df[months_df['demand'] > 0].copy()
    adi_months = np.mean(months_df.reset_index()[
                         'date'].diff())/np.timedelta64(1, 'M')
    cv_months = (np.std(months_df['demand'])/np.mean(months_df['demand']))**2
    cat_months = demand_cat(adi_months, cv_months)

    last_demand_date = temp_df.index.max()

    return (adi_days, cv_days, cat_days, adi_weeks, cv_weeks, cat_weeks, adi_months, cv_months, cat_months, last_demand_date)


def calc_leadtime_fl(df):

    df.loc[df['median_lt_days'] >= 26, 'lt_fl'] = 0.4
    df.loc[(df['median_lt_days'] < 26) & (
        df['median_lt_days'] >= 11), 'lt_fl'] = 0.6
    df.loc[df['median_lt_days'] < 11, 'lt_fl'] = 1

    return df


def demand_cat(adi, cv):

    if adi <= 1.32 and cv <= 0.49:
        return 'smooth'
    elif adi <= 1.32 and cv > 0.49:
        return 'erratic'
    elif adi > 1.32 and cv <= 0.49:
        return 'intermittent'
    elif adi > 1.32 and cv > 0.49:
        return 'lumpy'

# To create date related features for XGBoost Model


def xgb_create_date_features(df_actual, label=None):
    """
    Creates time series features from datetime index
    """

    df = df_actual.copy()
    cols = list(df.columns)

    df['DATE'] = df.index
    df['HOUR'] = df['DATE'].dt.hour
    df['DAYOFWEEK'] = df['DATE'].dt.dayofweek
    df['QUARTER'] = df['DATE'].dt.quarter
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year
    df['DAYOFYEAR'] = df['DATE'].dt.dayofyear
    df['DAYOFMONTH'] = df['DATE'].dt.day
    df['WEEKOFYEAR'] = df['DATE'].dt.weekofyear

    date_cols = ['HOUR', 'DAYOFWEEK', 'QUARTER', 'MONTH', 'YEAR',
                 'DAYOFYEAR', 'DAYOFMONTH', 'WEEKOFYEAR']

    if label:
        y = df[label]
        cols.remove(label[0])
        X = df[cols + date_cols]
        return X, y
    else:
        X = df[cols + date_cols]
        return X


# formatitng data for fbprophet
def format_fb(df):

    df = df.reset_index().copy()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    return df
