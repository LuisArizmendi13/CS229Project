import os

# Define the correct class order
new_class_order = ["unhealthy_leaf", "blossom", "unripe", "weed", "ripe"]

# Load the original classes.txt file
classes_file = "classes.txt"

if not os.path.exists(classes_file):
    print("Error: classes.txt not found.")
    exit()

with open(classes_file, "r") as f:
    old_classes = [line.strip() for line in f.readlines()]

# Create a mapping from old index to new index
class_mapping = {
    old_classes.index(cls): new_class_order.index(cls) for cls in old_classes
}

# Save the new class order in classes.txt
with open(classes_file, "w") as f:
    for cls in new_class_order:
        f.write(cls + "\n")

# Process each .txt label file (excluding classes.txt)
for filename in os.listdir():
    if filename.endswith(".txt") and filename != "classes.txt":
        with open(filename, "r") as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                old_index = int(parts[0])
                new_index = class_mapping.get(
                    old_index, old_index
                )  # Default to old_index if not found
                updated_line = f"{new_index} " + " ".join(parts[1:])
                updated_lines.append(updated_line)

        with open(filename, "w") as f:
            f.write("\n".join(updated_lines) + "\n")

print("Class indices updated successfully!")
