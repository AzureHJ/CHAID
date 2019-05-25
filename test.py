from CHAID.tree import Tree
from ChiMerge.bin_chi_merge import BinChiMerge
import numpy as np 
import pandas as pd

def map_function(x):  
    x = str(x)  
    try:  
        return np.float64(x.replace(",", ""))  
    except:  
        return np.nan 

bm = BinChiMerge()

data = pd.read_csv("CHAID_data.csv")  
test = data[["上年销售额", "hand_mark"]].copy() 
test["上年销售额"] = test["上年销售额"].map(lambda x: map_function(x)) 
test = test.dropna() 

bm.fit(test, "上年销售额", "hand_mark")
test1 = bm.transform(test, "上年销售额")   
tree = Tree.from_pandas_df(test1, {"上年销售额": "ordinal"}, "hand_mark", min_parent_node_size=100, min_child_node_size=50, split_threshold=3.84)
split_map = tree.tree_store[0].split.split_map 
result_interval = []
for split in split_map:
    result_interval.append([bm.mapping_dict_[split[0]][0], bm.mapping_dict_[split[-1]][1]])