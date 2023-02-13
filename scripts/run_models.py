import pandas as pd
from utils.ml import arima_select, fb_fit_model, ets_model, croston, croston_tsb, xgb_forecast
from utils.eval import evaluate_model
from utils.de_func import format_fb, xgb_create_date_features

def models_run(df, m, freq, test_periods, mat_code):

    metrics = []
    pred = []

    # SARIMA
    train, test, forecast = arima_select(df, m, test_periods)
    sarima_metrics, sarima_pred = evaluate_model(
        train, test, forecast, 'SARIMA', mat_code)

    metrics.append(sarima_metrics)
    pred.append(sarima_pred)

    # FBProphet
    # formatting data for fbprophet
    formatted_df = format_fb(df)
    train, test, model, forecast, forecast_plot = fb_fit_model(
        formatted_df, test_periods, freq)
    fb_metrics, fb_pred = evaluate_model(
        train, test, forecast_plot, 'FbProphet', mat_code)
    metrics.append(fb_metrics)
    pred.append(fb_pred)

    # Triple Exponential Smoothing
    damped = True
    train, test, forecast = ets_model(df, test_periods, m, damped)
    tes_metrics, tes_pred = evaluate_model(
        train, test, forecast, 'TES', mat_code)
    metrics.append(tes_metrics)
    pred.append(tes_pred)

    # Croston
    train, test, forecast = croston(df, test_periods)
    croston_metrics, croston_pred = evaluate_model(
        train, test, forecast, 'Croston', mat_code)
    metrics.append(croston_metrics)
    pred.append(croston_pred)

    # Croston TSB
    train, test, forecast = croston_tsb(df, test_periods)
    crostontsb_metrics, crostontsb_pred = evaluate_model(
        train, test, forecast, 'CrostonTSB', mat_code)
    metrics.append(crostontsb_metrics)
    pred.append(crostontsb_pred)

    # XGB
    train, test, forecast = xgb_forecast(df, test_periods)
    xgb_metrics, xgb_pred = evaluate_model(
        train, test, forecast, 'XGB', mat_code)
    metrics.append(xgb_metrics)
    pred.append(xgb_pred)

    metrics_df = pd.concat(metrics)
    pred_df = pd.concat(pred)

    # deleting model that is not needed
    del model

    return metrics_df, pred_df
