# -*- coding:utf-8 -*-

import warnings
import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time

def prediction_seasonal(data, seasonal_s = 12):
    q = d = range(0, 2)
    # 定义p参数以获取0到3之间的任何值
    p = range(0, 2)
    # 生成p，q和q的所有不同组合
    pdq = list(itertools.product(p, d, q))

    seasonal_pdq = [(x[0], x[1], x[2], seasonal_s) for x in pdq]
    print('季节性ARIMA的参数组合示例')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

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
    # 使用训练数据拟合模型
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=SARIMAX_model[a_i_c.index(min(a_i_c))][0],
                                    seasonal_order=SARIMAX_model[a_i_c.index(min(a_i_c))][1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results

def prediction_seasonal_pred(data, start, end, seasonal_s = 12):
    results = prediction_seasonal(data, seasonal_s)

    return results.get_prediction(start=start, end=end, dynamic=False)
