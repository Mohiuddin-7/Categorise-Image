import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import shutil
import sys

# Load class names dynamically from dataset folder
dataset_path = "categorized_images"
class_names = sorted(os.listdir(dataset_path))  # Get categories from folder names
num_classes = len(class_names)

# Load trained model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(1280, num_classes)
model.load_state_dict(torch.load("custom_image_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_and_categorize(image_path, move=True):
    """Predict category for an image and move it to the correct folder."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Error loading image: {e}")
        return

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    predicted_category = class_names[predicted_index]
    confidence = probabilities[0][predicted_index].item()

    print(f"✅ {image_path} -> **Predicted Category:** {predicted_category} ({confidence:.2%} confidence)")

    # Move image to categorized_images folder
    if move:
        category_folder = os.path.join("categorized_images", predicted_category)
        os.makedirs(category_folder, exist_ok=True)
        shutil.move(image_path, os.path.join(category_folder, os.path.basename(image_path)))
        print(f"📂 Moved to: {category_folder}\n")

def process_folder(folder_path):
    """Process all images in a folder."""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            predict_and_categorize(os.path.join(folder_path, file))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

        if os.path.isdir(input_path):
            print(f"\n📂 **Processing folder:** {input_path}\n")
            process_folder(input_path)
        elif os.path.isfile(input_path):
            predict_and_categorize(input_path)
        else:
            print("❌ Invalid path. Please provide an image or folder.")
    else:
        print("⚠️ Please provide an image or folder path.")
