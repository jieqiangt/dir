from pmdarima import auto_arima
from utils.de_func import xgb_create_date_features
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import sys
import warnings
import logging
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", DeprecationWarning)  # Turn off pystan warnings
warnings.simplefilter("ignore", FutureWarning)
logging.getLogger('fbprophet').setLevel(
    logging.ERROR)  # Turn off fbprophet stdout logger


# preventing printing
class NullWriter(object):
    def write(self, arg):
        pass


def arima_select(df, m, test_periods):

    nullwrite = NullWriter()
    oldstdout = sys.stdout

    sys.stdout = nullwrite  # disable print output

    stepwise = auto_arima(df['demand'], start_p=0, start_q=0,
                          max_p=2, max_q=2,  seasonal=True, m=m, max_P=2, max_Q=2,
                          trace=True, error_action='ignore', suppress_warnings=True,
                          stepwise=True, disp=False)

    # split data into train & test.
    # For weekly, periods = 8. Test set is last 2 months.
    # For monthly, periods = 3. Test set is last 3 months.
    train = df[:len(df)-test_periods]
    test = df[len(df)-test_periods:]

    # build model with best model chosen by auto arima using training data
    # use trend 'c' to enforce intercept for AR if autoarima suggests to use intercept
    if stepwise.with_intercept:
        model = SARIMAX(train['demand'], trend='c',
                        order=stepwise.order, seasonal_order=stepwise.seasonal_order,
                        )
    else:
        model = SARIMAX(train['demand'],
                        order=stepwise.order, seasonal_order=stepwise.seasonal_order)

    fitted_model = model.fit(disp=0)
    # fitted_model.summary()
    # Using trained model, get predict values
    forecast = fitted_model.get_forecast(len(test))

    predictions = pd.DataFrame(forecast.predicted_mean)
    predictions = predictions.join(forecast.conf_int(0.05))
    predictions.columns = ['forecast', 'lower_ci', 'upper_ci']
    predictions.index.name = 'date'
    # Clip lower_ci, forecast to 0 since negative values are not sensible
    predictions.loc[predictions['lower_ci'] < 0, 'lower_ci'] = 0
    predictions.loc[predictions['forecast'] < 0, 'forecast'] = 0

    sys.stdout = oldstdout  # enable output

    # deleting fitted model to save memory
    del fitted_model

    return train, test, predictions


def fb_fit_model(df, test_periods, freq='W'):

    train = df[:len(df)-test_periods]
    test = df[len(df)-test_periods:]

    model = Prophet(uncertainty_samples=100, interval_width=0.9)
    model.fit(train)

    future = model.make_future_dataframe(periods=test_periods, freq=freq)
    forecast = model.predict(future)

    # clipping forecast lower_ci & forecast to 0 if negative
    forecast.loc[forecast['yhat'] < 0, 'yhat'] = 0
    forecast.loc[forecast['yhat_lower'] < 0, 'yhat_lower'] = 0

    # changing to plotting format
    train.set_index('ds', inplace=True)
    train.columns = ['demand']
    test.set_index('ds', inplace=True)
    test.columns = ['demand']

    forecast_plot = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_plot.set_index('ds', inplace=True)
    forecast_plot.index.name = 'date'
    forecast_plot.columns = ['forecast', 'lower_ci', 'upper_ci']

    return train, test, model, forecast, forecast_plot


def croston(ts, test_periods=8, alpha=0.4):

    # storing timestamp index
    time_index = ts.index

    # splitting into train & test for plotting
    test = ts[ts.shape[0]-test_periods:]
    train = ts[:-test_periods]

    d = np.array(train)  # Transform the input into a numpy array
    cols = len(d)  # Historical period length
    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan]*test_periods)

    # level (a), periodicity(p) and forecast (f)
    a, p, f = np.full((3, cols+test_periods), np.nan)
    q = 1  # periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0]/p[0]
    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t]
            p[t+1] = alpha*q + (1-alpha)*p[t]
            f[t+1] = a[t+1]/p[t+1]
            q = 1
        else:
            a[t+1] = a[t]
            p[t+1] = p[t]
            f[t+1] = f[t]
            q += 1

    # Future Forecast
    a[cols+1:cols+test_periods] = a[cols]
    p[cols+1:cols+test_periods] = p[cols]
    f[cols+1:cols+test_periods] = f[cols]

    df = pd.DataFrame.from_dict(
        {"actual": d, "forecast": f, "period": p, "level": a, "date": time_index})
    df.set_index(time_index, inplace=True)
    df.drop(columns=['date'], inplace=True)

    # clipping negative forecast to 0
    df.loc[df['forecast'] < 0, 'forecast'] = 0

    # changing into plotting format
    forecast = df[['forecast']]

    test.index.name = 'date'
    forecast.index.name = 'date'

    return train, test, forecast


def croston_tsb(ts, test_periods=8, alpha=0.4, beta=0.4):

    # storing timestamp index
    time_index = ts.index

    # getting forecast period results
    test = ts[ts.shape[0]-test_periods:]
    train = ts[:-test_periods]

    d = np.array(train)  # Transform the input into a numpy array
    cols = len(d)  # Historical period length
    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan]*test_periods)

    # level (a), probability(p) and forecast (f)
    a, p, f = np.full((3, cols+test_periods), np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0]*a[0]

    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t]
            p[t+1] = beta*(1) + (1-beta)*p[t]
        else:
            a[t+1] = a[t]
            p[t+1] = (1-beta)*p[t]
        f[t+1] = p[t+1]*a[t+1]

    # Future Forecast
    a[cols+1:cols+test_periods] = a[cols]
    p[cols+1:cols+test_periods] = p[cols]
    f[cols+1:cols+test_periods] = f[cols]

    df = pd.DataFrame.from_dict(
        {"actual": d, "forecast": f, "period": p, "level": a, "date": time_index})
    df.set_index(time_index, inplace=True)
    df.drop(columns=['date'], inplace=True)

    # clipping negative forecast to 0
    df.loc[df['forecast'] < 0, 'forecast'] = 0

    # changing into plotting format
    forecast = df[['forecast']]

    test.index.name = 'date'
    forecast.index.name = 'date'

    return train, test, forecast


def ets_model(df, test_periods=8, m=52, damped=False):

    # split into train & test
    train = df[:-test_periods]
    test = df[df.shape[0] - test_periods:]

    # instantiate model
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=m, damped=damped)
    # fit model
    fit = model.fit()
    # obtain predictions
    pred = fit.forecast(test_periods)

    # formatting for plotting
    pred = pd.DataFrame(pred)
    pred.columns = ['forecast']

    pred.index.name = 'date'
    # clipping negative forecast to 0
    pred.loc[pred['forecast'] < 0, 'forecast'] = 0

    return train, test, pred


def xgb_forecast(df, test_periods):

    # split into train & test
    train = df[:-test_periods]
    test = df[df.shape[0] - test_periods:]

    # Creating date features for XGBoost model & output X_train, y_train, X_test, y_test
    X_train, y_train = xgb_create_date_features(train, ['demand'])
    X_test, y_test = xgb_create_date_features(test, ['demand'])

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, verbose=False)  # Change verbose to True if you want to see it train

    test['forecast'] = reg.predict(X_test)
    forecast = np.round(test[['forecast']])
    test.drop(columns=['forecast'], inplace=True)

    test.index.name = 'date'
    forecast.index.name = 'date'

    return train, test, forecast
