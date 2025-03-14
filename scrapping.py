import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm  # Progress bar

# Setup Chrome Driver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in background
options.add_argument("--disable-gpu")  # Prevents rendering issues
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open Pexels search page
search_url = "https://www.pexels.com/search/productivity/"
driver.get(search_url)

# Wait for images to load
time.sleep(5)

# Scroll down multiple times to load more images
for _ in range(10):  
    driver.execute_script("window.scrollBy(0, 2000);")
    time.sleep(2)  # Wait for new images to load

# Find all image elements
images = driver.find_elements(By.TAG_NAME, "img")

# Extract Image URLs
image_urls = []
for img in images:
    url = img.get_attribute("src")
    if url and "pexels.com" in url:  # Ensure it's a valid image link
        image_urls.append(url)

# Keep only the first 100 images
image_urls = image_urls[:100]

# Create folder if not exists
save_folder = "Productivity"
os.makedirs(save_folder, exist_ok=True)

# Download and save images
for idx, img_url in enumerate(tqdm(image_urls, desc="Downloading Images")):
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(save_folder, f"image_{idx+1}.jpg"), "wb") as f:
            f.write(img_data)
    except Exception as e:
        print(f"Error downloading image {idx+1}: {e}")

# Close the browser
driver.quit()

print(f"✅ {len(image_urls)} images downloaded in the '{save_folder}' folder.")
