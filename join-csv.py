import pandas as pd

# Define file paths
file1_path = 'path of first Dataset file'
file2_path = 'path of secound Dataset file'
output_path = 'path of joined Dataset file'

# Load the CSV files
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

df1.drop('label_y_x',inplace=True,axis=1)
df1.drop('label_x_x',inplace=True,axis=1)
df1.drop('notification_x_x',inplace=True,axis=1)
df1.drop('notification_y_x',inplace=True,axis=1)
# Ensure the 'timestamp' columns are in datetime format
df1['timestamp'] = pd.to_datetime(df1['timestamp'] ,format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
df2['timestamp'] = pd.to_datetime(df2['timestamp'] ,format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
# Perform an inner join on the 'timestamp' column
merged_df = pd.merge(df1,df2, on='timestamp', how='inner')

# Save the resulting DataFrame to a new CSV file
merged_df.to_csv(output_path, index=False)

print(f"Joined file saved to {output_path}")
