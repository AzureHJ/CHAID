import numpy as np
import pandas as pd
from collections import Counter 

def map_function(x): 
    x = str(x) 
    try: 
        return np.float64(x.replace(",", "")) 
    except: 
        return np.nan

data = pd.read_csv("CHAID_data.csv") 
test = data[["上年销售额", "hand_mark"]].copy()
test["上年销售额"] = test["上年销售额"].map(lambda x: map_function(x))
test = test.dropna() 
%time chi_merge(data=test, feature="上年销售额", label="hand_mark", max_intervals=100) 

def discretization_mapping(data, feature, mapping_intervals, inplace=False):
    if not inplace:
        data = data.copy()
    mapping_dict = {str(interval): \
                    data[data[feature].between(interval[0], interval[1])].index.values \
                    for interval in mapping_intervals}
    for mapping_string, mapping_index in mapping_dict.items():
        data.loc[mapping_index, feature] = mapping_string
    if not inplace:
        return data