import pandas as pd

# Load the DOOR.csv file
door_df = pd.read_csv('G:/term5/dev/Events/Cam_Event/cam.csv')

# Filter records by DOOR value
win_open = door_df[door_df['privacy'] == 1 ]
win_close = door_df[door_df['privacy'] == 0 ]

# Save filtered data to separate CSV files
win_open.to_csv('G:/term5/dev/Events/Cam_Event/cam_on.csv', index=False)
win_close.to_csv('G:/term5/dev/Events/Cam_Event/cam_off.csv', index=False)

print("Records with DOOR value 1 saved to open_door.csv")
print("Records with DOOR value 0 saved to close_door.csv")
