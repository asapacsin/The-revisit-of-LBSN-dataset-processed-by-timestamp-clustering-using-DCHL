import pickle
import pandas as pd

def inspect_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type of object: {type(data)}")
    
    if isinstance(data, pd.DataFrame):
        print("First 5 rows:")
        print(data.head())
        print("\nShape:", data.shape)
        print("\nColumns:", data.columns.tolist())
        print("\nInfo:")
        print(data.info())
    elif isinstance(data, list):
        print("First 5 elements:")
        for item in data[:5]:
            print(item)
        print("\nNumber of elements:", len(data))
    elif isinstance(data, dict):
        print("Keys:", list(data.keys()))
        for key, value in data.items():
            print(f"\nFirst 5 values for '{key}':")
            if isinstance(value, (list, pd.Series)):
                print(value[:5])
            elif isinstance(value, pd.DataFrame):
                print(value.head())
            else:
                print(value)
    else:
        print("Object type not specifically handled. Printing repr:")
        print(repr(data))

# Example usage
inspect_pkl('output.pkl')