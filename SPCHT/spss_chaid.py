from ChiMerge.bin_chi_merge import BinChiMerge
from CHAID.tree import Tree
import pandas as pd
import numpy as np

class SPCHT:
    """
    SPSS中的CHAID算法是可以处理连续型变量的。
    当自变量是连续型变量时，先对其卡方分箱为有序的离散变量，再
    使用CHAID算法构建决策树。
    TODO: 卡方分箱目前只支持因变量为2分类变量的情况。
          由于评分卡的特点，这个类目前也只支持输入单个变量的情况。
          只支持pandas dataframe作为输入。
    """

    def __init__(self, analyze_df, feature, label, chi_merge_significance=0.05, 
                chi_merge_max_intervals=1,  chaid_alpha_merge=0.05, chaid_max_depth=2, 
                chaid_min_parent_node_size=100, chaid_min_child_node_size=50, chaid_split_threshold=0,
                chaid_weight=None, chaid_dep_variable_type='categorical'):
        self.data = analyze_df
        self.chi_merge = BinChiMerge(chi_merge_significance, chi_merge_max_intervals)
        self.chaid_tree = Tree.from_pandas_df(self.data, feature, label, chaid_alpha_merge,
                        chaid_max_depth, chaid_min_parent_node_size, chaid_min_child_node_size, 
                        chaid_split_threshold, chaid_weight, chaid_dep_variable_type)
