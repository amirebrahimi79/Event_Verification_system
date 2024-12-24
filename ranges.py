import pandas as pd
from datetime import timedelta

# Load the DOOR.csv file
door_df = pd.read_csv('path of event record .csv file', parse_dates=['timestamp'])

# Prepare a list to store formatted lines
output_lines = []

# Iterate through each row to create the intervals
for _, row in door_df.iterrows():
    t = row['timestamp']
    
    # Define each interval based on t
    intervals = [
        (t + timedelta(seconds=10), t + timedelta(seconds=15), 0, 'Non-Event'),
        (t + timedelta(seconds=15), t+ timedelta(seconds=20), 1, 'Spoofing-Attack'),
        (t, t + timedelta(seconds=5), 0, 'Masking-Attack'),
        (t + timedelta(seconds=5), t + timedelta(seconds=10), 1, 'Event')
    ]
    
    # Format each interval and add to the list
    for interval in intervals:
        output_lines.append(f"('{interval[0]}', '{interval[1]}', {interval[2]}, '{interval[3]}'),")
        #("4/4/2019  4:36:18 AM", "4/4/2019 9:27:30 AM", 1),
# Write all lines to the output.txt file
with open('path of output ranges file.txt', 'w') as file:
    file.write("\n".join(output_lines))

print("Intervals saved to output.txt")
