import os
from pillow_heif import register_heif_opener
from PIL import Image

# Register HEIC support in PIL
register_heif_opener()

# Define input HEIC folder and output JPG folder
heic_folder = "strawberrys"  # Change this to your actual HEIC folder path
output_folder = os.path.join(heic_folder, "converted_jpgs")  # Saves converted images here

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Convert HEIC to JPG properly
for filename in os.listdir(heic_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(heic_folder, filename)
        jpg_filename = filename.replace(".HEIC", ".jpg").replace(".heic", ".jpg")  # Ensure correct replacement
        jpg_path = os.path.join(output_folder, jpg_filename)

        try:
            img = Image.open(heic_path)
            img = img.convert("RGB")  # Convert to RGB to prevent format issues
            img.save(jpg_path, "JPEG", quality=95)  # Save as JPG with high quality

            print(f"✅ Converted: {filename} → {jpg_filename}")

        except Exception as e:
            print(f"❌ Error converting {filename}: {e}")

print(f"\n🚀 All HEIC images successfully converted to JPG in '{output_folder}'!")
