import warnings
import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import generate.line as gline
import generate.prediction_seasonal as prediction_seasonal
import generate.prediction_random as prediction_random
plt.style.use('fivethirtyeight')

data = gline.sample_line("2018-01-01 10:40:30", total=96*2, initial=10.0, slope=-0.1, rdm=5.0)

data = pd.DataFrame(data)
data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
data.set_index(['time'], inplace=True)

#pred = prediction_seasonal.prediction_seasonal_pred(data, data.index[-12], pd.to_datetime('2018-01-11 09:40:30'))
pred = prediction_random.prediction_random_pred(data, data.index[-12], pd.to_datetime('2018-01-11 09:40:30'))
pred_ci = pred.conf_int()

ax = data.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

plt.show()
