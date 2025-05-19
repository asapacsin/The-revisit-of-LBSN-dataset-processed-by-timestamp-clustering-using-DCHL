# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import pickle

def load_list_with_pkl(filename):
    try:
        with open(filename, 'rb') as f:
            list_obj = pickle.load(f)
        return list_obj
    except Exception as e:
        print(f"Error loading pickle file {filename}: {e}")
        raise

# Load data
data_filename = 'train_poi_zero.txt'
try:
    data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
    sessions_dict = data[0]  # poiID starts from 0
    labels_dict = data[1]
except Exception as e:
    print(f"Error processing data: {e}")
    raise

# Print dictionaries and their lengths
print(f'session: {sessions_dict}')
print(f'session len: {len(sessions_dict)}')
print(f'labels: {labels_dict}')
print(f'labels len: {len(labels_dict)}')

# Find maximum key
if not sessions_dict:
    print("Error: sessions_dict is empty")
elif not labels_dict:
    print("Error: labels_dict is empty")
else:
    max_session_key = max(sessions_dict.keys())
    max_label_key = max(labels_dict.keys())
    print(f'Maximum key in sessions_dict: {max_session_key}')
    print(f'Maximum key in labels_dict: {max_label_key}')
    
    # Verify key consistency
    if max_session_key != max_label_key:
        print("Warning: Maximum keys differ between sessions_dict and labels_dict")
    if set(sessions_dict.keys()) != set(labels_dict.keys()):
        print("Warning: Keys in sessions_dict and labels_dict do not match")