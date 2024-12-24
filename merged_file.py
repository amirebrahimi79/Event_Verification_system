import pandas as pd
import os

# Specify the folder containing the .csv files
folder_path = 'dircetory path of sensor folder'  # Replace with your folder path
output_file = 'path of merged file'     # Name for the merged output file
timestamp_column = 'timestamp'         # Replace with the name of your timestamp column
# List to hold DataFrames
dataframes = []

# Loop through all .csv files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # Convert the timestamp column to datetime format with specified format
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        
        # Append DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Remove rows with invalid timestamp (NaT)
merged_df.dropna(subset=[timestamp_column], inplace=True)

# Sort by timestamp
merged_df.sort_values(by=timestamp_column, inplace=True)

# Remove duplicate rows based on the timestamp column
merged_df.drop_duplicates(subset=[timestamp_column], inplace=True)

# Save the merged DataFrame to a new .csv file
merged_df.to_csv(output_file, index=False)

print(f'Merged {len(dataframes)} files and removed duplicates based on {timestamp_column} into {output_file}')

