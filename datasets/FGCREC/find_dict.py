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

def check_users(sessions_dict, labels_dict, user_ids, dict_name):
    """Check for specified user IDs and display their sessions/labels"""
    print(f"\nChecking users in {dict_name}:")
    for user_id in user_ids:
        if user_id in sessions_dict:
            print(f"User {user_id} found in {dict_name} sessions_dict: {sessions_dict[user_id]}")
        else:
            print(f"User {user_id} missing from {dict_name} sessions_dict")
        if user_id in labels_dict:
            print(f"User {user_id} found in {dict_name} labels_dict: {labels_dict[user_id]}")
        else:
            print(f"User {user_id} missing from {dict_name} labels_dict")
        print()

# Load train data
train_filename = 'train_poi_zero.pkl'
try:
    train_data = load_list_with_pkl(train_filename)
    train_sessions_dict = train_data[0]
    train_labels_dict = train_data[1]
except Exception as e:
    print(f"Error processing train data: {e}")
    raise

# Load test data
test_filename = 'test_poi_zero.pkl'
try:
    test_data = load_list_with_pkl(test_filename)
    test_sessions_dict = test_data[0]
    test_labels_dict = test_data[1]
except Exception as e:
    print(f"Error processing test data: {e}")
    raise

# Users to check
missing_users = [27, 1765, 4776, 4920, 5124, 5180, 5496, 5538, 5626]

# Check users in train set
check_users(train_sessions_dict, train_labels_dict, missing_users, "train")

# Check users in test set
check_users(test_sessions_dict, test_labels_dict, missing_users, "test")

# Check for user 2771
print("\nChecking user 2771:")
if 2771 in train_sessions_dict:
    print(f"User 2771 found in train sessions_dict: {train_sessions_dict[2771]}")
else:
    print("Warning: User 2771 missing from train sessions_dict")
if 2771 in train_labels_dict:
    print(f"User 2771 found in train labels_dict: {train_labels_dict[2771]}")
else:
    print("Warning: User 2771 missing from train labels_dict")
if 2771 in test_sessions_dict:
    print(f"User 2771 found in test sessions_dict: {test_sessions_dict[2771]}")
else:
    print("Warning: User 2771 missing from test sessions_dict")
if 2771 in test_labels_dict:
    print(f"User 2771 found in test labels_dict: {test_labels_dict[2771]}")
else:
    print("Warning: User 2771 missing from test labels_dict")

# Print summary statistics
print(f"\nTrain sessions_dict len: {len(train_sessions_dict)}")
print(f"Train labels_dict len: {len(train_labels_dict)}")
print(f"Test sessions_dict len: {len(test_sessions_dict)}")
print(f"Test labels_dict len: {len(test_labels_dict)}")

# Verify key consistency
if train_sessions_dict.keys() != train_labels_dict.keys():
    missing_in_train_labels = set(train_sessions_dict.keys()) - set(train_labels_dict.keys())
    missing_in_train_sessions = set(train_labels_dict.keys()) - set(train_sessions_dict.keys())
    print(f"\nWarning: Train keys mismatch")
    if missing_in_train_labels:
        print(f"User IDs in train sessions_dict but missing in labels_dict: {sorted(missing_in_train_labels)}")
    if missing_in_train_sessions:
        print(f"User IDs in train labels_dict but missing in sessions_dict: {sorted(missing_in_train_sessions)}")
if test_sessions_dict.keys() != test_labels_dict.keys():
    missing_in_test_labels = set(test_sessions_dict.keys()) - set(test_labels_dict.keys())
    missing_in_test_sessions = set(test_labels_dict.keys()) - set(test_sessions_dict.keys())
    print(f"\nWarning: Test keys mismatch")
    if missing_in_test_labels:
        print(f"User IDs in test sessions_dict but missing in labels_dict: {sorted(missing_in_test_labels)}")
    if missing_in_test_sessions:
        print(f"User IDs in test labels_dict but missing in sessions_dict: {sorted(missing_in_test_sessions)}")