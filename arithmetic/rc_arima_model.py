# -*- coding:utf-8 -*-

import time
import warnings
import itertools
import pandas as pd
import statsmodels.api as sm
import datetime


class RcArimaModel(object):

    def __init__(self, data=[], seasonal_s=12):
        self.data =data
        self.pdq = []
        self.seasonal_pdq = []
        self.seasonal_s = seasonal_s
        self.train_data_end_time = None
        self.forecast_data_end_time = None

    def load_data(self, data):
        self.data = data
        print('训练数据截止时间：' + self.data.time.values[-1])
        self.train_data_end_time = datetime.datetime.strptime(self.data.time.values[-1], '%Y-%m-%d %H:%M:%S')

    def prepare_data(self):
        # 数据集预处理
        self.data['time'] = pd.to_datetime(self.data['time'], format='%Y-%m-%d %H:%M:%S')
        self.data.set_index(['time'], inplace=True)

    def prepare_pdq(self):
        # 定义d和q参数以获取0到1之间的任何值
        q = d = range(0, 2)
        # 定义p参数以获取0到3之间的任何值
        p = range(0, 4)
        # 生成p，q和q的所有不同组合
        self.pdq = list(itertools.product(p, d, q))

    def prepare_season_pdq(self):
        # 生成季节性p，q和q三元组的所有不同组合,这里s=12,每2小时间隔的监控采集值
        self.seasonal_pdq = [(x[0], x[1], x[2], self.seasonal_s) for x in self.pdq]
        print('季节性ARIMA的参数组合示例')
        print('SARIMAX: {} x {}'.format(self.pdq[1], self.seasonal_pdq[1]))
        print('SARIMAX: {} x {}'.format(self.pdq[1], self.seasonal_pdq[2]))
        print('SARIMAX: {} x {}'.format(self.pdq[2], self.seasonal_pdq[3]))
        print('SARIMAX: {} x {}'.format(self.pdq[2], self.seasonal_pdq[4]))

    # duration 代表持续预测时长 单位是小时
    def train_and_forcast(self, duration):
        # 忽略警告消息
        warnings.filterwarnings("ignore")
        # 计时器开启
        start = time.time()

        # 计算AIC参数, 根据训练数据集的大小和参数范围选择耗时不同。
        a_i_c = []
        SARIMAX_model = []
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.data,
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
        mod = sm.tsa.statespace.SARIMAX(self.data,
                                        order=SARIMAX_model[a_i_c.index(min(a_i_c))][0],
                                        seasonal_order=SARIMAX_model[a_i_c.index(min(a_i_c))][1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        results = mod.fit(disp=False)

        # 进行预测
        # 这里需要算出截止时间点
        self.forecast_data_end_time = self.train_data_end_time + datetime.timedelta(hours=duration)
        end_time_str = self.forecast_data_end_time.strftime('%Y-%m-%d %H:%M:%S')
        print("预测数据截止时间:"+end_time_str)
        pred = results.get_forecast(end_time_str)
        pred_ci = pred.conf_int()

        return pred

    def execute(self, duration):
        if self.data.empty:
            print('无训练数据！')
        else:
            self.prepare_data()
            self.prepare_pdq()
            self.prepare_season_pdq()
            return self.train_and_forcast(duration)

