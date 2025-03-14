import os
import pandas as pd
import requests
from tqdm import tqdm

# Load the CSV file
csv_file = "insparation.csv"  # Make sure this is the correct file name
df = pd.read_csv(csv_file)
print("Column Names in CSV:", df.columns.tolist())

# Ensure the column name matches your file
url_column = "Image-link"  # Change this if the column name is different

# Destination folder
save_folder = "Motivation"
os.makedirs(save_folder, exist_ok=True)

# Set limit to 80 images
num_images = min(80, len(df))  # If there are less than 80 URLs, take all available

# Download images
for idx, url in tqdm(enumerate(df[url_column][:num_images]), total=num_images):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image_path = os.path.join(save_folder, f"motivation_{idx+1}.jpg")
            with open(image_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print(f"Downloaded {num_images} images to {save_folder}")
