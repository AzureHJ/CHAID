import numpy as np
import pandas as pd

def binary_chi_merge(data, feature, label, significance=0.05, max_intervals=1):
    """
    二分类问题的简化版本，仅支持label为0和1的情况。
    经过计算优化，可读性低
    """
    assert data[label].drop_duplicates().shape[0] == 2 \
        and 1 in data[label].drop_duplicates().values \
        and 0 in data[label].drop_duplicates().values 

    # 获得对应显著性水平的卡方临界值
    # TODO: 确定临界值, 临时填充自由度为1, 显著性水平为0.05的临界值
    chi_threshold = 3.84
    # 排序
    distinct_vals = sorted(set(data[feature]))
    # 初始化区间, [x, y),[y, z)
    intervals = [[ele, ele] for ele in distinct_vals]
    for i in range(1, len(distinct_vals) - 1):
        intervals[i][1] = intervals[i+1][0]
    # 排序
    data = data.sort_values(by=feature).reset_index(drop=True)
    # 构建初始化 区间-分布 字典, 用于避免重复计算
    interval_distribution_dict = {}
    for interval in intervals:
        total_count = data.loc[(data[feature] >= interval[0]) & (data[feature] < interval[1])].shape[0]
        ones_count = data.loc[(data[feature] >= interval[0]) & (data[feature] < interval[1]), label].sum()
        zeros_count = total_count - ones_count
        interval_distribution_dict[str(interval)] = np.array([zeros_count, ones_count])
    while len(intervals) > max_intervals: 
        chi_list = []
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
            # 计算卡方值
            chi = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
            # 处理0的情况
            chi_list.append(np.sum(chi)) 
        # 将卡方值小于临界值的合并, 否则结束分箱
        if np.min(chi_list) > chi_threshold:
            break
        min_chi_index = np.argmin(chi_list)
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

def map_function(x): 
    x = str(x) 
    try: 
        return np.float64(x.replace(",", "")) 
    except: 
        return np.nan

if __name__ == "__main__":
    data = pd.read_csv("CHAID_data.csv") 
    test = data[["上年销售额", "hand_mark"]].copy()
    test["上年销售额"] = test["上年销售额"].map(lambda x: map_function(x))
    test = test.dropna() 
    print(binary_chi_merge(data=test, feature="上年销售额", label="hand_mark", max_intervals=100))