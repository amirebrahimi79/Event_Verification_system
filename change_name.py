import os
# D:/Amir/term4/Test/Event_Window/Development/pi1/RSS
# Directory containing your files
directory = "path of the directory"  # Change this to the path of your directory

# Iterate through all files in the directory
for filename in os.listdir(directory):
        # Split filename and extension
        name, ext = os.path.splitext(filename)
        
        # Check if the file name has at least 12 characters (excluding extension)
        
        # Create the new name by taking the first 10 characters (YYYYMMDDHH)
        new_name = name[:10] + ext
            
        # Create full paths for renaming
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
            
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")

print("Renaming complete.")
