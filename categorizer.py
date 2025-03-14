import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load class names dynamically
dataset_path = "categorized_images"
class_names = sorted(os.listdir(dataset_path))  # Get categories from folder names
num_classes = len(class_names)

# Load trained model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(1280, num_classes)
model.load_state_dict(torch.load("custom_image_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_category(image_path):
    """Predicts the category of a single image."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
    
    return class_names[predicted_index]

def categorize_images(image_folder="uncategorized_images", output_folder="categorized_images"):
    """Categorizes all images in a folder."""
    if not os.path.exists(image_folder):
        print("❌ Image folder not found!")
        return

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if not os.path.isfile(img_path):
            continue
        
        category = predict_category(img_path)
        category_folder = os.path.join(output_folder, category)
        os.makedirs(category_folder, exist_ok=True)
        
        new_path = os.path.join(category_folder, img_name)
        os.rename(img_path, new_path)
        print(f"✅ Moved {img_name} to {category}/")

if __name__ == "__main__":
    categorize_images()
    print("✅ Categorization complete!")
