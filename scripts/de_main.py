from de_calc import hist_lead_time_calc, latest_lead_time_calc, hist_demand_purchase_calc, demand_profile_cat, soh_calc, inv_backcalc, dues_out_calc, safety_stock_calc
from utils.de_generic import consolidate_df
import pandas as pd


def de_main():

    hist_lt_df = hist_lead_time_calc()
    latest_lt_df = latest_lead_time_calc(hist_lt_df)
    ts_df = hist_demand_purchase_calc()
    cat_df = demand_profile_cat(ts_df)

    materials = ts_df['material'].unique()
    cat_df = cat_df['material'].isin(materials)

    soh_df = soh_calc(materials)
    inv_df = inv_backcalc(ts_df, soh_df)
    do_df = dues_out_calc()
    safety_stock_df = safety_stock_calc(inv_df, cat_df)

    cat_df = cat_df['material'].isin(materials)
    do_df = do_df['material'].isin(materials)

    ts_df = consolidate_df(ts_df, safety_stock_df)
    ts_df['date'] = pd.to_datetime(ts_df['date'], format='%Y-%m-%d')
    cat_df = consolidate_df(cat_df, latest_lt_df)

    ts_df.to_csv("./data/cleaned/ts_df.csv", index=False)
    cat_df.to_csv("./data/cleaned/cat_df.csv", index=False)
    do_df.to_csv("./data/cleaned/do_df.csv", index=False)
