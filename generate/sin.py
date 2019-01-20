# -*- coding:utf-8 -*-

import math
import random
import time
import datetime

'''
正弦函数采样生成器
@start 开始时间
@interval 时间间隔
@total 采样总数
@frequency 周期频率
@flex  值伸缩
@interval 初始值
@slope 斜率
@rdm 随机数
'''
def sample_sin(start, interval=3600, total=200, frequency=24, flex=1.0, initial=0.0, slope=0.0, rdm=0.0):
    startSeconds = time.mktime(time.strptime(start, "%Y-%m-%d %H:%M:%S"))
    samples = []
    for x in range(total):
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(startSeconds + x*interval))
        samples.append({
            'time': dt,
            'value': math.sin((math.pi/frequency)*x) * flex + initial + (slope*x) + random.uniform(0, rdm)
        })

    return samples


if __name__ == '__main__':
    print('sin...')
    a = sample_sin("2018-01-01 10:40:30", total=96, initial=0.0, slope=0.1, rdm=1.0)
    print(a)