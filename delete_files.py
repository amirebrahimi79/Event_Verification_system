import os

# D:\Amir\term4\Start\DOOR
bmp_folder = 'Sensor Folder path'
door_folder = 'Event Folder path'

# Get the list of filenames in both folders (excluding extensions)
bmp_files = {os.path.splitext(f)[0] for f in os.listdir(bmp_folder) if os.path.isfile(os.path.join(bmp_folder, f))}
door_files = {os.path.splitext(f)[0] for f in os.listdir(door_folder) if os.path.isfile(os.path.join(door_folder, f))}

# Find files in BMP that are not in DOOR
uncommon_files = bmp_files - door_files

# Delete uncommon files in the BMP folder
for file in uncommon_files:
    full_path = os.path.join(bmp_folder, file)
    for ext in ['.csv' , '.CSV']:  # Adjust for file extension if needed
        if os.path.exists(full_path + ext):
            os.remove(full_path + ext)
            print(f"Deleted '{full_path + ext}'")

print("Uncommon file deletion complete.")
