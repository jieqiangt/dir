import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima  # for determining ARIMA orders

import sys

import warnings
import logging
warnings.filterwarnings("ignore")
# Turn off pystan warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
# Turn off fbprophet stdout logger
logging.getLogger('fbprophet').setLevel(logging.ERROR)


# preventing printing
class NullWriter(object):
    def write(self, arg):
        pass

# Finding SARIMA p,d,q values


def arima_select(df, m, test_periods):

    nullwrite = NullWriter()
    oldstdout = sys.stdout

    sys.stdout = nullwrite  # disable print output

    stepwise = auto_arima(df['demand'], start_p=0, start_q=0,
                          max_p=2, max_q=2,  seasonal=True, m=m, max_P=2, max_Q=2,
                          trace=True, error_action='ignore', suppress_warnings=True,
                          stepwise=True, disp=False)

    # print(f'Total number of rows: {df.shape[0]}')

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
