import numpy as np
import pandas as pd
from scipy.stats import chi2

class BinChiMerge:
    """
    二分类问题的简化版本，仅支持label为0和1的情况。
    """
    def __init__(self, significance=0.05, max_intervals=1):
        self.significance_ = significance
        self.max_intervals_ = max_intervals
        self.intervals_ = None

    def fit(self, data, feature, label):
        self.intervals_ = self._fit(data, feature, label)
        return self.intervals_

    def transform(self, data, feature, inplace=False):
        if self.intervals_ is None:
            return None
        else:
            return self._transform(data, feature, self.intervals_, inplace)

    def _fit(self, data, feature, label):
        assert data[label].drop_duplicates().shape[0] == 2 \
        and 1 in data[label].drop_duplicates().values \
        and 0 in data[label].drop_duplicates().values 

        # 排序
        distinct_vals = sorted(set(data[feature]))
        # 初始化区间, [x, y),[y, z)
        intervals = [[ele, ele] for ele in distinct_vals]
        for i in range(1, len(distinct_vals) - 1):
            intervals[i][1] = intervals[i+1][0]
        # 卡方阈值列表
        chi_threshold_list = []
        # 构建初始化 区间-分布 字典, 用于避免重复计算
        interval_distribution_dict = {}
        for interval in intervals:
            total_count = data.loc[(data[feature] >= interval[0]) & (data[feature] < interval[1])].shape[0]
            ones_count = data.loc[(data[feature] >= interval[0]) & (data[feature] < interval[1]), label].sum()
            zeros_count = total_count - ones_count
            interval_distribution_dict[str(interval)] = np.array([zeros_count, ones_count])
        while len(intervals) > self.max_intervals_: 
            chi_list = []
            chi_threshold_list = []
            # 计算每两个区间之间的卡方值
            for i in range(len(intervals)-1):
                """
                变量说明
                ----------------+---+---+----
                interval/label  | 0 | 1 | 
                ----------------+---+---+----
                interval_1      | a | b |
                ----------------+---+---+----
                interval_2      | c | d |
                ----------------+---+---+----
                """
                a = interval_distribution_dict[str(intervals[i])][0]
                b = interval_distribution_dict[str(intervals[i])][1]
                c = interval_distribution_dict[str(intervals[i + 1])][0]
                d = interval_distribution_dict[str(intervals[i + 1])][1]
                n = a + b + c + d
                # 获得卡方值的阈值，自由度为n-1
                chi_threshold = chi2.ppf(1 - self.significance_, n - 1)
                chi_threshold_list.append(chi_threshold)
                # 计算卡方值
                chi = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
                chi_list.append(np.sum(chi)) 
            # 将卡方值小于临界值的合并, 否则结束分箱
            # TODO: 一次扫描仅合并两个区间是低效的
            min_chi_index = np.argmin(chi_list)
            if np.min(chi_list) > chi_threshold_list[min_chi_index]:
                break
            interval_1 = intervals[min_chi_index]
            interval_2 = intervals[min_chi_index + 1]
            combined_interval = [interval_1[0], interval_2[1]]
            combined_distribution = interval_distribution_dict[str(interval_1)] + interval_distribution_dict[str(interval_2)]
            del intervals[min_chi_index: min_chi_index+2]
            del interval_distribution_dict[str(interval_1)]
            del interval_distribution_dict[str(interval_2)]
            interval_distribution_dict[str(combined_interval)] = combined_distribution
            intervals.insert(min_chi_index, combined_interval)
        
        return intervals

    def _transform(self, data, feature, intervals, inplace=False):
        if not inplace:
            data = data.copy()
        mapping_dict = {str(interval): \
                    data[data[feature].between(interval[0], interval[1])].index.values \
                    for interval in intervals}
        for mapping_string, mapping_index in mapping_dict.items():
            data.loc[mapping_index, feature] = mapping_string
        return data

    def fit_transform(self, data, feature, label, inplace=False):
        return self._transform(data, feature, self._fit(data, feature, label), inplace)

    