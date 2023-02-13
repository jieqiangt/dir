import pandas as pd
import numpy as np
from utils.de_generic import resample_df, format_time_series


def ml_main():
    
    # reading in data files
    df = pd.read_csv('./cleaned/ts_df.csv')
    meta_df = pd.read_csv('./cleaned/cat_df.csv')

    # some material codes are numeric type Code to check: df[df['mat_code'].str.isnumeric().isna()]
    # converting to string type
    df["mat_code"] = df["mat_code"].astype("str")

    # getting unique material codes that are filtered in meta_df
    mat_codes = sorted(meta_df['mat_code'].unique())

    # filtering for only material codes for modelling
    df = df.loc[np.isin(df['mat_code'], mat_codes), :]

    # declaring arguments for modelling
    freq_str = 'monthly'
    backtest = True

    freq_dict = {'weekly': {'m': 52,
                            'freq': 'W',
                            'test_periods': 8},
                'monthly': {'m': 12,
                            'freq': 'M',
                            'test_periods': 3}}

    m = freq_dict[freq_str]['m']
    freq = freq_dict[freq_str]['freq']
    test_periods = freq_dict[freq_str]['test_periods']

    num_codes = 100
    total = len(df['mat_code'].unique())
    print(f'Total number of mat codes: {total}')
    max_splits = round(total/num_codes)
    print(f'Total number of splits: {max_splits}')

    # split into 100 codes for each run due to memory limitations
    for run_split in range(0, max_splits):

        start_index = run_split * num_codes

        if run_split == max_splits-1:
            end_index = total
        else:
            end_index = (run_split + 1) * num_codes

        print(f'Running split: {run_split}')
        print(f'Running forecast for {start_index}:{end_index}')
        run_codes = mat_codes[start_index:end_index]

        pred_store = []
        metrics_store = []

        i = 0

        for mat_code in run_codes:

            try:
                # Formatting df into time series index
                base_df = format_time_series(df, mat_code)
                base_df = base_df.resample(rule=freq).sum()[['demand']]

                metrics_df, pred_df = models_run(
                    base_df, m, freq, test_periods, mat_code)

                # storing results
                metrics_store.append(metrics_df)
                pred_store.append(pred_df)
                i = i + 1
                print(f'{i} forecasting completed for: {mat_code}')

            except:
                print(f'Error encountered: {mat_code}')
                continue

        print(f'Split {run_split} Completed...')

        pred_output = pd.concat(pred_store)
        metrics_output = pd.concat(metrics_store)

        pred_output.to_csv(
            f'./output/pred/backtest/{freq_str}_pred_backtest_output_{str(run_split)}.csv', index=False)
        metrics_output.to_csv(
            f'./output/metrics/backtest/{freq_str}_metrics_backtest_output_{str(run_split)}.csv', index=False)
