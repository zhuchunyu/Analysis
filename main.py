# -*- coding:utf-8 -*-

from arithmetic.rc_arima_model import RcArimaModel
import pandas as pd
import warnings
import json


if __name__ == '__main__':
    arima = RcArimaModel(seasonal_s=12)
    data = pd.read_csv('cpu_data.csv', engine='python', skipfooter=0)
    arima.load_data(data)
    result = arima.execute(50)
    print(result.predicted_mean['2018-12-17 00:00:00':'2018-12-27 22:00:00'])
    # with open("monitor.json", "r") as f:
    #     monitor_contents = json.load(f)
    #     if monitor_contents:
    #         for data in monitor_contents.get('data', []):
    #             times = []
    #             value = []
    #             train_data = {'time': times, 'value': value}
    #             print("指标名称"+data.get('commonMetricName'))
    #             for item in data.get('metricTimestampValues'):
    #                 if item.get('time', '') not in times and item.get('time', '') >'2018-12-31':
    #                     value.append(item.get('average', 0))
    #                     times.append(item.get('time', ''))
    #             train_data_frame = pd.DataFrame(train_data)
    #             # 目前每小时一条数据所以seasonal_s为24
    #             rc_arima = RcArimaModel(seasonal_s=0)
    #             rc_arima.load_data(train_data_frame)
    #             result = rc_arima.execute(10*12)
    #
    #             print(result.predicted_mean[0:])