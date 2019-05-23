import numpy as np
import pandas as pd
from collections import Counter

def chi_merge(data, feature, label, max_intervals):
    # 排序
    distinct_vals = sorted(set(data[feature]))
    # 获得所有可能的label
    labels = sorted(set(data[label])) 
    # 用于统计计算中每个label出现的次数
    empty_count = {l: 0 for l in labels} 
    # 初始化区间, [x, y)
    intervals = [[ele, ele] for ele in distinct_vals]
    for i in range(1, len(distinct_vals) - 1):
        intervals[i][1] = intervals[i+1][0]
    while len(intervals) > max_intervals: 
        chi = []
        # 计算每两个区间之间的卡方值
        for i in range(len(intervals)-1):
            # 计算卡方值
            first_interval_observated = data.loc[(data[feature] >= intervals[i][0]) & (data[feature] < intervals[i][1])]
            second_interval_observated = data.loc[(data[feature] >= intervals[i+1][0]) & (data[feature] < intervals[i+1][1])]
            # 两个区间内一共有多少个数据
            total = len(first_interval_observated) + len(second_interval_observated)
            # 每个类在不同的区间内出现的次数
            first_interval_count = np.array([v for i, v in {**empty_count, **Counter(first_interval_observated[label])}.items()])
            second_interval_count = np.array([v for i, v in {**empty_count, **Counter(second_interval_observated[label])}.items()])
            total_count = first_interval_count + second_interval_count
            first_interval_expected = total_count * (sum(first_interval_count) / total)
            second_interval_expected = total_count * (sum(second_interval_count) / total)
            # 计算卡方值
            chi_ = (first_interval_count - first_interval_expected) ** 2 / first_interval_expected + \
                (second_interval_count - second_interval_expected) ** 2 / second_interval_expected
            # 处理0的情况
            chi_ = np.nan_to_num(chi_) 
            chi.append(sum(chi_)) 
        # 将卡方值最小的两个区间合并
        min_chi_index = np.argmin(chi)
        tmp = intervals[min_chi_index] + intervals[min_chi_index+1]
        del intervals[min_chi_index: min_chi_index+2]
        intervals.insert(min_chi_index, [min(tmp), max(tmp)])
    return intervals