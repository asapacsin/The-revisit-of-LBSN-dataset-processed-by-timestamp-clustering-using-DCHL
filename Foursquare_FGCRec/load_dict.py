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

data_filename = 'test_poi_zero.pkl'
data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
sessions_dict = data[0]  # poiID starts from 0
labels_dict = data[1]
# print(f'session:{sessions_dict}')
print(f'len session:{len(sessions_dict)}')
# print(f'labels:{labels_dict}')
print(f'len label:{len(labels_dict)}')