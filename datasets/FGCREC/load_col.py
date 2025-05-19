# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import pickle
def load_list_with_pkl(filename):
    with open(filename, 'rb') as f:
        list_obj = pickle.load(f)

    return list_obj

data_filename = 'FGCREC_pois_coos_poi_zero.pkl'
data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
col = data[0]  
print(f'col:{col}')
print(f'col len:{len(data)}')