import os

def count_files_in_folders(base_path):
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            print(f"Folder: {folder_name}, File Count: {file_count}")

# Example usage:
base_path = '/Volumes/T7 Shield/Viton/train'
count_files_in_folders(base_path)
