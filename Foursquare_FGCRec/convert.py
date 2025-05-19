import pandas as pd
import pickle

# Initialize the output structure
data = [{}, {}]  # data[0] for sessions_dict, data[1] for labels_dict

# Step 1: Load the test data
test_data_path = 'Foursquare_test.txt'
try:
    test_data = pd.read_csv(test_data_path, sep='\s+', header=None, 
                            names=['group_id', 'session_id', 'label'], 
                            na_values=['', 'NA', 'nan', 'NaN'])
except Exception as e:
    print(f"Error reading file: {e}")
    raise

# Step 2: Process data to group sessions
sessions_dict = {}
skipped_rows = []

for index, row in test_data.iterrows():
    # Check for NaN or invalid values
    if pd.isna(row['group_id']) or pd.isna(row['session_id']) or pd.isna(row['label']):
        skipped_rows.append(index)
        continue
    
    try:
        # Convert to appropriate types
        group_id = int(row['group_id'])
        session_id = int(row['session_id'])
        
        # Add to sessions_dict
        if group_id not in sessions_dict:
            sessions_dict[group_id] = []
        sessions_dict[group_id].append(session_id)
        
    except (ValueError, TypeError) as e:
        print(f"Invalid data at row {index}: {e}")
        skipped_rows.append(index)
        continue

# Step 3: Create sessions (excluding last session_id, as nested list) and labels
labels_dict = {}
for group_id, session_ids in list(sessions_dict.items()):
    # Require at least 2 session_ids (1 for session, 1 for label)
    if len(session_ids) < 2:
        print(f"Warning: Group {group_id} has too few session_ids ({len(session_ids)})")
        skipped_rows.append(f"Group {group_id}")
        del sessions_dict[group_id]
        continue
    
    # Session excludes the last session_id, wrapped in a list
    sessions_dict[group_id] = [session_ids[:-1]]
    # Label is the last session_id
    labels_dict[group_id] = session_ids[-1]

# Step 4: Assign to data structure
data[0] = sessions_dict
data[1] = labels_dict

# Step 5: Save data to a pickle file in binary mode
try:
    with open('test_poi_zero.txt', 'wb') as f:
        pickle.dump(data, f)
except Exception as e:
    print(f"Error saving pickle file: {e}")
    raise

# Step 6: Verify and report
print("Sessions dictionary:", data[0])
print("Labels dictionary:", data[1])
if skipped_rows:
    print(f"Skipped rows or groups due to invalid data: {skipped_rows}")
if not sessions_dict or not labels_dict:
    print("Warning: Empty dictionaries. Check input data for issues.")