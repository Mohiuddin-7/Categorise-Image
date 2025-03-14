import requests
import os

# List of image URLs (Replace this with actual image links)
image_urls = [
    "https://example.com/image1.jpg",  # Replace with real image URLs
    "https://example.com/image2.jpg",
    # Add more image URLs here
]

# Folder to save images
save_folder = "categorized_images/News"
os.makedirs(save_folder, exist_ok=True)

# Function to download images
def download_image(url, folder):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

        # Check if response is an image
        if "image" in response.headers["Content-Type"]:
            filename = os.path.join(folder, url.split("/")[-1])
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"✅ Downloaded: {filename}")
        else:
            print(f"❌ Not an image: {url}")

    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")

# Download only the first 80 images
for index, url in enumerate(image_urls[:80]):  # Limit to 80 images
    download_image(url, save_folder)

print("\n🎉 Done! Downloaded up to 80 images.")
