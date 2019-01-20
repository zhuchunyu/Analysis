# -*- coding:utf-8 -*-

import warnings
import itertools
import statsmodels.api as sm
import time

def prediction_random(data):
    q = d = range(0, 4)
    # 定义p参数以获取0到3之间的任何值
    p = range(0, 4)
    # 生成p，q和q的所有不同组合
    pdq = list(itertools.product(p, d, q))
    print(len(pdq))

    warnings.filterwarnings("ignore")
    # 计时器开启
    start = time.time()

    # 计算AIC参数, 根据训练数据集的大小和参数范围选择耗时不同。
    a_i_c = []
    SARIMAX_model = []
    for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                            full_output=False)
            results = mod.fit(disp=False)

            print('SARIMAX {} - AIC: {}'.format(param, results.aic))
            a_i_c.append(results.aic)
            SARIMAX_model.append([param])
        except Exception as err:
            print(err)
            continue

    # 计算耗时
    print('%.2f sec' % (time.time() - start))
    print('最小 AIC 值为: {} 对应模型参数: SARIMAX{}'.format(min(a_i_c), SARIMAX_model[a_i_c.index(min(a_i_c))][0]))
    # 使用训练数据拟合模型
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=SARIMAX_model[a_i_c.index(min(a_i_c))][0],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit(disp=False)

    return results

def prediction_random_pred(data, start, end):
    results = prediction_random(data)

    return results.get_prediction(start=start, end=end, dynamic=False)
