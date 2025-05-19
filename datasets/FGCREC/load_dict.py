# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import pickle
import os

def load_list_with_pkl(filename):
    """Load a pickle file and validate its structure"""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        if os.path.getsize(filename) == 0:
            raise ValueError(f"File {filename} is empty")
        if not filename.endswith('.pkl'):
            raise ValueError(f"File {filename} is not a pickle file (expected .pkl extension)")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Validate structure
        if not isinstance(data, list) or len(data) != 2:
            raise ValueError(f"Invalid pickle structure in {filename}: expected [sessions_dict, labels_dict]")
        if not isinstance(data[0], dict) or not isinstance(data[1], dict):
            raise ValueError(f"Invalid data types in {filename}: sessions_dict and labels_dict must be dictionaries")
        
        return data
    except Exception as e:
        print(f"Error loading pickle file {filename}: {e}")
        raise

# Load data
data_filename = 'test_poi_zero.pkl'
try:
    data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
    sessions_dict = data[0]  # poiID starts from 0
    labels_dict = data[1]
except Exception as e:
    print(f"Error processing data: {e}")
    raise

# Print dictionaries and their lengths
print(f'\nsession: {sessions_dict}')
print(f'session len: {len(sessions_dict)}')
print(f'labels: {labels_dict}')
print(f'labels len: {len(labels_dict)}')

# Check for user 2771
if 2771 in sessions_dict:
    print(f"User 2771 found in sessions_dict with sessions: {sessions_dict[2771]}")
else:
    print("Warning: User 2771 missing from sessions_dict")
if 2771 in labels_dict:
    print(f"User 2771 found in labels_dict with label: {labels_dict[2771]}")
else:
    print("Warning: User 2771 missing from labels_dict")

# Identify missing user IDs
sessions_keys = set(sessions_dict.keys())
labels_keys = set(labels_dict.keys())
missing_in_labels = sessions_keys - labels_keys
missing_in_sessions = labels_keys - sessions_keys

print(f"\nMissing user IDs:")
if missing_in_labels:
    print(f"User IDs in sessions_dict but missing in labels_dict: {sorted(missing_in_labels)}")
else:
    print("No user IDs missing in labels_dict")
if missing_in_sessions:
    print(f"User IDs in labels_dict but missing in sessions_dict: {sorted(missing_in_sessions)}")
else:
    print("No user IDs missing in sessions_dict")

# Find maximum key
if not sessions_dict:
    print("Error: sessions_dict is empty")
elif not labels_dict:
    print("Error: labels_dict is empty")
else:
    max_session_key = max(sessions_dict.keys())
    max_label_key = max(labels_dict.keys())
    print(f'\nMaximum key in sessions_dict: {max_session_key}')
    print(f'Maximum key in labels_dict: {max_label_key}')
    
    # Verify key consistency
    if max_session_key != max_label_key:
        print("Warning: Maximum keys differ between sessions_dict and labels_dict")
    if sessions_keys != labels_keys:
        print("Warning: Keys in sessions_dict and labels_dict do not match")