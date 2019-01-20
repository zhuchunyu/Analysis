# -*- coding:utf-8 -*-

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

data = pd.read_csv('csv/amount.csv', engine='python', skipfooter=0)

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data.set_index(['date'], inplace=True)

y = data

print(y)

# y['2018-11-02':].plot(figsize=(15, 6))
# plt.show()

p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore")  # specify to ignore warning messages

a_i_c = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

            a_i_c.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=SARIMAX_model[a_i_c.index(min(a_i_c))][0],
                                seasonal_order=SARIMAX_model[a_i_c.index(min(a_i_c))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()


pred = results.get_prediction(start=y.index[-2], end=pd.to_datetime('2019-01-31'), dynamic=False)
pred_ci = pred.conf_int()

ax = y.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.1)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

plt.show()