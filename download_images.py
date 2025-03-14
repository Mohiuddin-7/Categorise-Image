import pandas as pd
import requests
import os

file_path = "open-images-dataset-train0.tsv"

# Read TSV file, skipping the first row
df = pd.read_csv(file_path, sep="\t", engine="python", skiprows=1, names=["ImageURL", "Subset", "ImageID"])

# Print first few rows to verify
print("First few rows of the cleaned dataset:")
print(df.head())

# Create a fixed category folder (since 'Subset' contains numbers, not real categories)
output_folder = "open_images_v7/dataset"
os.makedirs(output_folder, exist_ok=True)

# Limit downloads to the first 100 images
max_images = 100

for index, row in df.iterrows():
    if index >= max_images:
        break  # Stop downloading after 100 images

    image_url = row["ImageURL"]
    image_id = row["ImageID"]

    # Ensure the image filename ends with ".jpg"
    image_path = os.path.join(output_folder, f"{image_id}.jpg")

    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded: {image_id}.jpg")
        else:
            print(f"❌ Failed: {image_id}")
    except Exception as e:
        print(f"❌ Error downloading {image_id}: {e}")
