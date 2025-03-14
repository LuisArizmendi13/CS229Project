import os
import shutil
import random

# Define paths
source_folder = "./Year 2 plants"  # Update with actual path
train_folder = "./Year 2 plants/training"
val_folder = "./Year 2 plants/validation"

# Create folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get all JPEG files (paired with .txt)
files = [
    f for f in os.listdir(source_folder) if f.endswith(".jpeg") or f.endswith(".jpg")
]
base_names = list(set(f.split(".")[0] for f in files))  # Get unique base names

# Shuffle and split (80% train, 20% val)
random.shuffle(base_names)
split_idx = int(0.8 * len(base_names))
train_set = base_names[:split_idx]
val_set = base_names[split_idx:]


# Function to move paired files
def move_files(file_list, destination):
    for base in file_list:
        jpeg_file = os.path.join(source_folder, base + ".jpeg")
        jpg_file = os.path.join(source_folder, base + ".jpg")
        txt_file = os.path.join(source_folder, base + ".txt")

        for file in [jpeg_file, jpg_file, txt_file]:
            if os.path.exists(file):
                shutil.move(file, os.path.join(destination, os.path.basename(file)))


# Move files
move_files(train_set, train_folder)
move_files(val_set, val_folder)

print("Dataset successfully split into training and validation folders!")
