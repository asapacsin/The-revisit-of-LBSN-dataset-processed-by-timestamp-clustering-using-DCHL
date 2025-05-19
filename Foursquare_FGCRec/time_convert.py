import pandas as pd
import pickle
import math
from collections import defaultdict
import os
import io

# Configuration
DATA_PATH = 'Foursquare_checkins.txt'
TEST_SIZE = 0.2
RANDOM_STATE = 42
SESSION_TIMEOUT = 6 * 60 * 60  # 6 hours in seconds (adjust to 2 * 60 * 60 for more sessions)
MAX_USER_ID = 7461

def load_and_preprocess():
    """Load data, filter by user_id, and show statistics"""
    try:
        df = pd.read_csv(DATA_PATH, sep='\s+', header=None,
                        names=['user_id', 'poi_id', 'timestamp'])
    except Exception as e:
        print(f"Error reading file {DATA_PATH}: {e}")
        raise
    
    # Initial statistics
    initial_users = df['user_id'].nunique()
    print(f"\nInitial data:")
    print(f"Rows (check-ins): {len(df)}")
    print(f"Unique POIs: {df['poi_id'].nunique()}")
    print(f"Unique users: {initial_users}")
    
    # Filter by user_id
    pre_filter_users = set(df['user_id'].unique())
    df = df[df['user_id'] <= MAX_USER_ID]
    filtered_users = set(df['user_id'].unique())
    
    # Post-filtering statistics
    print(f"\nAfter filtering user_id <= {MAX_USER_ID}:")
    print(f"Rows (check-ins): {len(df)}")
    print(f"Unique POIs: {df['poi_id'].nunique()}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Users removed by filter: {sorted(pre_filter_users - filtered_users)}")
    
    return df.sort_values(['user_id', 'timestamp']), initial_users

def create_sessions(df):
    """Group check-ins into sessions with realistic timeout"""
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff()
    df['new_session'] = (
        (df['time_diff'] > SESSION_TIMEOUT) |
        (df['user_id'] != df['user_id'].shift()))
    df['session_id'] = df.groupby('user_id')['new_session'].cumsum()
    
    # Debug user 2771
    if 2771 in df['user_id'].values:
        user_2771_sessions = df[df['user_id'] == 2771].groupby('session_id').agg({
            'poi_id': list,
            'timestamp': ['count', 'min', 'max']
        })
        print(f"\nUser 2771 session details: {user_2771_sessions.to_dict()}")
    else:
        print("\nUser 2771 not found in data after filtering")
    
    # Report session counts
    session_counts = df.groupby('user_id')['session_id'].nunique()
    print(f"\nSession stats per user:\n{session_counts.describe()}")
    print(f"Users with sessions: {len(session_counts)}")
    return df.drop(columns=['time_diff', 'new_session'])

def split_and_label_sessions(df):
    """Split sessions and create labels from last POIs, ensuring test set includes users"""
    user_sessions = defaultdict(list)
    for user_id, group in df.groupby('user_id'):
        sessions = group.groupby('session_id')['poi_id'].apply(list).tolist()
        user_sessions[user_id] = sessions
    
    train_sessions = defaultdict(list)
    test_sessions = defaultdict(list)
    train_labels_dict = {}
    test_labels_dict = {}
    
    for user_id, sessions in user_sessions.items():
        if not sessions or not any(s for s in sessions):
            continue  # Skip users with no valid sessions
        
        # Calculate split index
        split_idx = max(1, math.floor(len(sessions) * (1 - TEST_SIZE)))
        
        # If last test session has only one POI, include one more session
        if split_idx < len(sessions) and len(sessions[split_idx]) == 1 and split_idx > 1:
            split_idx -= 1
            print(f"Adjusted split for user {user_id}: last test session had one POI, moved split_idx to {split_idx}")
        
        # Split sessions
        train = sessions[:split_idx]
        test = sessions[split_idx:]
        
        # Get label from last POI of last training session
        if train and train[-1]:
            train_labels_dict[user_id] = int(train[-1][-1])
            train[-1] = train[-1][:-1]
            if not train[-1]:
                train = train[:-1]
        
        # Get label from last POI of last test session
        if test and test[-1]:
            test_labels_dict[user_id] = int(test[-1][-1])
            test[-1] = test[-1][:-1]
            if not test[-1]:
                test = test[:-1]
        
        # Only include users with non-empty sessions
        if train:
            train_sessions[user_id] = train
        if test:
            test_sessions[user_id] = test
    
    return dict(train_sessions), dict(test_sessions), train_labels_dict, test_labels_dict

def create_output_files(train_data, test_data, train_labels_dict, test_labels_dict):
    """Generate pickle and text output files with rigorous validation and load testing"""
    # Validate data structure and types
    for name, data, labels in [
        ('train', train_data, train_labels_dict),
        ('test', test_data, test_labels_dict)
    ]:
        if not isinstance(data, dict) or not isinstance(labels, dict):
            raise ValueError(f"Invalid {name} data or labels: must be dictionaries")
        if not data or not labels:
            print(f"Warning: {name} data or labels are empty")
        
        for user_id, sessions in data.items():
            if not isinstance(user_id, int):
                raise ValueError(f"Invalid user_id in {name}_data: {user_id} is not an integer")
            if not isinstance(sessions, list):
                raise ValueError(f"Invalid sessions for user {user_id} in {name}_data: not a list")
            for session in sessions:
                if not isinstance(session, list):
                    raise ValueError(f"Invalid session for user {user_id} in {name}_data: not a list")
                for poi_id in session:
                    if not isinstance(poi_id, int):
                        raise ValueError(f"Invalid poi_id {poi_id} for user {user_id} in {name}_data: not an integer")
        
        for user_id, label in labels.items():
            if not isinstance(user_id, int):
                raise ValueError(f"Invalid user_id in {name}_labels: {user_id} is not an integer")
            if not isinstance(label, int):
                raise ValueError(f"Invalid label {label} for user {user_id} in {name}_labels: not an integer")
        
        if data.keys() != labels.keys():
            print(f"Warning: {name} data and labels have mismatched keys")
            common_keys = set(data.keys()) & set(labels.keys())
            data = {k: data[k] for k in common_keys}
            labels = {k: labels[k] for k in common_keys}
            if not common_keys:
                print(f"Warning: No common keys in {name} data and labels")
    
    # Test serializability
    for name, data in [
        ('train', [train_data, train_labels_dict]),
        ('test', [test_data, test_labels_dict])
    ]:
        try:
            temp_buffer = io.BytesIO()
            pickle.dump(data, temp_buffer, protocol=3)
            temp_buffer.seek(0)
            pickle.load(temp_buffer)
            temp_buffer.close()
        except Exception as e:
            raise ValueError(f"Data for {name} is not serializable: {e}")
    
    # Save pickle files
    pickle_files = [
        ('train_poi_zero.pkl', [train_data, train_labels_dict]),
        ('test_poi_zero.pkl', [test_data, test_labels_dict])
    ]
    for filename, data in pickle_files:
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, protocol=3)
            file_size = os.path.getsize(filename)
            print(f"Saved {filename} (size: {file_size} bytes)")
            with open(filename, 'rb') as f:
                loaded_data = pickle.load(f)
            if not isinstance(loaded_data, list) or len(loaded_data) != 2:
                raise ValueError(f"{filename} has invalid structure: expected [sessions_dict, labels_dict]")
            if not isinstance(loaded_data[0], dict) or not isinstance(loaded_data[1], dict):
                raise ValueError(f"{filename} has invalid data: sessions_dict or labels_dict not dictionaries")
            print(f"Verified {filename} is readable")
            sample_sessions = dict(list(loaded_data[0].items())[:2])
            sample_labels = dict(list(loaded_data[1].items())[:2])
            print(f"Sample sessions from {filename}: {sample_sessions}")
            print(f"Sample labels from {filename}: {sample_labels}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            raise
    
    # Save text files
    text_files = [
        ('train_poi_zero.txt', train_data),
        ('test_poi_zero.txt', test_data)
    ]
    for filename, data in text_files:
        try:
            with open(filename, 'w') as f:
                for user_id, sessions in data.items():
                    for session_idx, session in enumerate(sessions):
                        for poi_id in session:
                            f.write(f"{user_id} {session_idx} {poi_id}\n")
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            raise

def main():
    # 1. Load and preprocess
    df, initial_users = load_and_preprocess()
    
    # 2. Create sessions
    df = create_sessions(df)
    filtered_users = df['user_id'].nunique()
    print(f"After session creation: {len(df)} check-ins from {filtered_users} users")
    
    # 3. Split sessions and create labels
    train_sessions, test_sessions, train_labels_dict, test_labels_dict = split_and_label_sessions(df)
    
    # User comparison
    train_users = len(train_sessions)
    test_users = len(test_sessions)
    print(f"\nUser comparison:")
    print(f"Original users (initial): {initial_users}")
    print(f"Users after filtering user_id <= {MAX_USER_ID}: {filtered_users}")
    print(f"Users in train set: {train_users}")
    print(f"Users in test set: {test_users}")
    print(f"Users lost (initial to train): {initial_users - train_users}")
    print(f"Users lost (filtered to train): {filtered_users - train_users}")
    print(f"Users lost (initial to test): {initial_users - test_users}")
    print(f"Users lost (filtered to test): {filtered_users - test_users}")
    
    # Debug: Print sample data
    print(f"\nTrain sessions sample: {dict(list(train_sessions.items())[:2])}")
    print(f"Train labels sample: {dict(list(train_labels_dict.items())[:2])}")
    print(f"Test sessions sample: {dict(list(test_sessions.items())[:2])}")
    print(f"Test labels sample: {dict(list(test_labels_dict.items())[:2])}")
    
    # 4. Create output files
    create_output_files(train_sessions, test_sessions, train_labels_dict, test_labels_dict)
    
    # 5. Print statistics
    train_counts = sum(len(s) for s in train_sessions.values())
    test_counts = sum(len(s) for s in test_sessions.values())
    print(f"\nFinal split:")
    print(f"Train sessions: {train_counts}")
    print(f"Test sessions: {test_counts}")
    print(f"Train/test ratio: {test_counts/(train_counts+test_counts):.1%}" if train_counts + test_counts > 0 else "Train/test ratio: N/A")
    print(f"Train labels created: {len(train_labels_dict)}")
    print(f"Test labels created: {len(test_labels_dict)}")

if __name__ == "__main__":
    main()