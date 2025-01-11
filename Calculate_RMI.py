import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

def calculate_rmi(csv_file, event_column, sensor_columns):
    """
    Calculate Relative Mutual Information (RMI) for each sensor column in the dataset.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - event_column (str): Name of the column containing event labels.
    - sensor_columns (list): List of column names containing sensor data.

    Returns:
    - rmi_dict (dict): Dictionary with sensor column names as keys and their RMI values as values.
    """
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Ensure the event column is categorical
    df[event_column] = df[event_column].astype('category')
    
    # Calculate the entropy of the event labels
    event_probs = df[event_column].value_counts(normalize=True)
    event_entropy = entropy(event_probs, base=2)
    
    # Initialize a dictionary to store RMI results
    rmi_dict = {}

    # Compute RMI for each sensor column
    for sensor in sensor_columns:
        # Mutual information between the event column and the sensor column
        mi = mutual_info_score(df[event_column], df[sensor])
        
        # Relative Mutual Information
        rmi = mi / event_entropy if event_entropy > 0 else 0
        rmi_dict[sensor] = rmi

    return rmi_dict


# Example Usage
if __name__ == "__main__":
    # Path to the CSV file
    csv_file = "example.csv"  # Replace with your CSV file path

    # Column names
    event_column = "event"  # Replace with the name of your event column
    sensor_columns = ["column1", "column2", "column3"]  # Replace with your sensor columns

    # Calculate RMI
    rmi_results = calculate_rmi(csv_file, event_column, sensor_columns)

    # Print RMI values
    print("Relative Mutual Information (RMI) for each sensor:")
    for sensor, rmi in rmi_results.items():
        print(f"{sensor}: {rmi:.4f}")
