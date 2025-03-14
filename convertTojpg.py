import os
from pillow_heif import register_heif_opener
from PIL import Image

register_heif_opener()

heic_folder = "strawberrys"
output_folder = os.path.join(heic_folder, "converted_jpgs")

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(heic_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(heic_folder, filename)
        jpg_filename = filename.replace(".HEIC", ".jpg").replace(".heic", ".jpg")
        jpg_path = os.path.join(output_folder, jpg_filename)
        
        try:
            img = Image.open(heic_path)
            img = img.convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)
            print(f"Converted: {filename} → {jpg_filename}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print(f"All HEIC images successfully converted to JPG in '{output_folder}'")
