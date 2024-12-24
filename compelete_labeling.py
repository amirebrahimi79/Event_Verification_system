import os
import pandas as pd

def filter_and_label_records_in_directory(input_dir, output_dir, ranges):
    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):  # Check if the file is a CSV
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)  # Save the output in the same filename but in the output directory
            
            # Load the CSV file
            df = pd.read_csv(input_file)

            # Convert the 'timestamp' column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f+00:00', errors='coerce')

            # Create new columns for notification and label
            df['notification'] = None
            df['label'] = None

            # Apply the label for each range in the specified 'ranges'
            for start_time_str, end_time_str, notifi_value, label_value in ranges:
                start_time = pd.to_datetime(start_time_str)
                end_time = pd.to_datetime(end_time_str)
                
                # Update the notification and label columns based on the timestamp ranges
                df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'notification'] = notifi_value
                df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'label'] = label_value

            # Save the modified records to a new CSV file in the output directory
            df.to_csv(output_file, index=False)
            print(f"Filtered and labeled records saved to {output_file}")

# Example usage:
input_dir = 'path of input directory'  # Directory containing your input CSV files
output_dir = 'path of input directory'  # Directory where you want to save the output CSV files

# Define the ranges: (start_time, end_time, notifi_value, label_value)
ranges = [
 'input ranges of timestamp and notification and label'
]

filter_and_label_records_in_directory(input_dir, output_dir, ranges)
