from PIL import Image
import os

folder_path = "categorized_images"  # Update with your dataset folder

for root, _, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify if it's a valid image
        except Exception as e:
            print(f"Corrupt image found: {file_path}, Error: {e}")
            os.remove(file_path)  # Remove corrupt image
            print(f"Deleted: {file_path}")
