# -*- coding:utf-8 -*-

import math
import random
import time
import datetime

def sample_line(start, interval=3600, total=200, initial=0.0, slope=0.0, rdm=0.0):
    startSeconds = time.mktime(time.strptime(start, "%Y-%m-%d %H:%M:%S"))
    samples = []
    for x in range(total):
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(startSeconds + x * interval))
        samples.append({
            'time': dt,
            'value': initial + (slope * x) + random.uniform(0, rdm)
        })

    return samples

if __name__ == '__main__':
    a = sample_line('2018-01-01 10:40:30', initial=10, total=50, slope=0.1, rdm=5.0)
    print(a)