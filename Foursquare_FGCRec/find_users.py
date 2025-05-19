import pandas as pd

# Load the data
data_path = 'Foursquare_checkins.txt'
df = pd.read_csv(data_path, sep='\s+', header=None, 
                 names=['user_id', 'poi_id', 'timestamp'])

print("=== Initial Data Check ===")
print(f"Total check-ins: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"First few user IDs: {sorted(df['user_id'].unique())[:10]}")

# Filter users with group_id > 7461
max_valid_group_id = 7461
filtered_df = df[df['user_id'] <= max_valid_group_id]

print("\n=== After Filtering ===")
print(f"Total check-ins remaining: {len(filtered_df)}")
print(f"Unique users remaining: {filtered_df['user_id'].nunique()}")
print(f"Users removed: {set(df['user_id'].unique()) - set(filtered_df['user_id'].unique())}")

# Check specific users
users_to_check = [0, 1, 2771]  # Add any other users you're concerned about
print("\n=== Specific User Check ===")
for user in users_to_check:
    user_data = filtered_df[filtered_df['user_id'] == user]
    print(f"User {user}: {len(user_data)} check-ins")

# Session assignment check
print("\n=== Session Assignment Check ===")
sample_user = 0  # Change to any user you want to examine
user_data = filtered_df[filtered_df['user_id'] == sample_user].sort_values('timestamp')
print(f"Check-ins for user {sample_user}:")
print(user_data.head(10))
print(len(df))