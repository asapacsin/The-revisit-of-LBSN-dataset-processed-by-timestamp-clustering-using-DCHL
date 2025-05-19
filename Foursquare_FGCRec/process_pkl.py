import pickle

def txt_to_pkl(input_txt_path, output_pkl_path):
    # Initialize dictionary to store index: [latitude, longitude]
    coord_dict = {}
    
    # Read the txt file
    with open(input_txt_path, 'r') as f:
        for line in f:
            # Split line by tabs
            index, lat, lon = line.strip().split('\t')
            # Convert index to int and coordinates to float
            index = int(index)
            lat = float(lat)
            lon = float(lon)
            # Store in dictionary
            coord_dict[index] = [lat, lon]
    
    # Save dictionary to pkl file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(coord_dict, f)
    
    print(f"Successfully converted {input_txt_path} to {output_pkl_path}")
    print("Sample dictionary contents:", {k: coord_dict[k] for k in list(coord_dict)[:5]})

# Example usage
txt_to_pkl('Foursquare_poi_coos.txt', 'output.pkl')