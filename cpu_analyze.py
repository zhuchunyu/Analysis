import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import time
plt.style.use('fivethirtyeight')

data = pd.read_csv('csv/cpu.csv', engine='python', skipfooter=0)

train_data_end_time = datetime.datetime.strptime(data.time.values[-1], '%Y-%m-%d %H:%M:%S')

data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
data.set_index(['time'], inplace=True)

#data = data.resample('6H').mean()

#data.plot(figsize=(15, 6))
#plt.show()

q = d = range(0, 2)
# 定义p参数以获取0到3之间的任何值
p = range(0, 2)
# 生成p，q和q的所有不同组合
pdq = list(itertools.product(p, d, q))

print(pdq)

seasonal_s = 12

seasonal_pdq = [(x[0], x[1], x[2], seasonal_s) for x in pdq]
print('季节性ARIMA的参数组合示例')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

print(seasonal_pdq)

warnings.filterwarnings("ignore")
# 计时器开启
start = time.time()

# 计算AIC参数, 根据训练数据集的大小和参数范围选择耗时不同。
a_i_c = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                            full_output=False)
            results = mod.fit(disp=False)

            print('SARIMAX {} x {} - AIC: {}'.format(param, param_seasonal, results.aic))
            a_i_c.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except Exception as err:
            print(err)
            continue

# 计算耗时
print('%.2f sec' % (time.time() - start))
print('最小 AIC 值为: {} 对应模型参数: SARIMAX{}x{}'.format(min(a_i_c), SARIMAX_model[a_i_c.index(min(a_i_c))][0],
                                                  SARIMAX_model[a_i_c.index(min(a_i_c))][1]))

print(a_i_c)
print(SARIMAX_model)
print([a_i_c.index(min(a_i_c))][0])
print(SARIMAX_model[a_i_c.index(min(a_i_c))][1])

# 使用训练数据拟合模型
mod = sm.tsa.statespace.SARIMAX(data,
                                order=SARIMAX_model[a_i_c.index(min(a_i_c))][0],
                                seasonal_order=SARIMAX_model[a_i_c.index(min(a_i_c))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
print(mod)

results = mod.fit(disp=False)

print(results.summary().tables[1])

#results.plot_diagnostics(figsize=(15, 12))
#plt.show()

pred = results.get_prediction(start=data.index[-50], end=pd.to_datetime('2019-01-17 20:00:00'), dynamic=False)
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
