import pandas as pd
import os

# Directory containing CSV files
input_directory = "G:/term5/dev/Events/Light_Event/light_on/RSS7"  # Replace with your directory path
output_directory = "G:/term5/dev/Events/Light_Event/light_on/RSS7"  # Directory to save cleaned files

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process each CSV file in the directory
for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):  # Ensure the file is a CSV
        input_file = os.path.join(input_directory, file_name)
        output_file = os.path.join(output_directory, file_name)
        
        # Read the CSV into a DataFrame
        df = pd.read_csv(input_file)
        
        # Remove rows where 'label' column has null values
        df_cleaned = df.dropna(subset=['label'])
        
        # Save the cleaned DataFrame back to a CSV file
        df_cleaned.to_csv(output_file, index=False)
        print(f"Processed and cleaned: {file_name}")

print(f"All files processed. Cleaned files are saved in '{output_directory}'.")
