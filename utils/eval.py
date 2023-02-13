

import pandas as pd
import numpy as np
from prophet.plot import plot_cross_validation_metric
from prophet.utilities import regressor_coefficients
from xgboost import plot_importance, plot_tree
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tools.eval_measures import rmse


def evaluate_model(train, test, forecast, model, material, plot_periods=15, plot_flag=False):

    df = pd.concat([train, test])

    # Calculating metrics
    eval_df = test.join(forecast)
    eval_df['forecast_bias'] = eval_df['forecast'] - eval_df['demand']
    eval_df['mad'] = abs(eval_df['forecast_bias'])
    eval_df['mape'] = eval_df['mad'] / eval_df['demand']
    rmse_error = rmse(eval_df['demand'], eval_df['forecast'])

    metrics_dict = {'forecast_bias': round(sum(eval_df['forecast_bias']), 1),
                    'forecast_bias_%': round(sum(eval_df['forecast']/sum(eval_df['demand']))*100, 1),
                    'mad': round(np.mean(eval_df['mad']), 1),
                    'mape': round(np.mean(eval_df['mape'] * 100), 1),
                    'rmse': round(rmse_error, 2),
                    'mean_po_qty': eval_df["demand"].mean(),
                    'total_po_qty': eval_df["demand"].sum()}

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    metrics_df = metrics_df.T
    metrics_df['model'] = model
    metrics_df['material'] = material

    # getting predicted output in required long format
    pred_output = forecast.copy()
    pred_output['model'] = model
    pred_output['material'] = material
    pred_output.reset_index(inplace=True)

    if plot_flag == True:

        print(f'Plotting results for {model}')
        # plot graph of test period using fitted model
        # Plot predictions against test values
        title = f'Demand Prediction {model}'
        ylabel = 'Quantity'
        xlabel = ''

        forecast_plot = forecast[len(forecast) - plot_periods:]

        ax = df[len(df) - plot_periods:]['demand'].plot(legend=True,
                                                        figsize=(12, 6), title=title)
        forecast_plot['forecast'].plot(legend=True)

        # plot upper CI & lower CI only if available
        if forecast_plot.shape[1] == 3:
            ax.fill_between(forecast_plot.index, (forecast_plot['lower_ci']), (
                forecast_plot['upper_ci']), color='tab:orange', alpha=.1)

        ax.autoscale(axis='x', tight=True)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(labels=['Actual', f'{model} forecast'])

    return metrics_df, pred_output
