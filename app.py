import streamlit as st
import os
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
# Set up dataset path
DATASET_PATH = "categorized_images"
os.makedirs(DATASET_PATH, exist_ok=True)

# Load class names dynamically from dataset folder
class_names = sorted(os.listdir(DATASET_PATH))  # Get categories from folder names
num_classes = len(class_names)

# Load the trained model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(1280, num_classes)
model.load_state_dict(torch.load("custom_image_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_and_save(image, filename):
    """Predict category and save the image in the correct folder."""
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
    
    predicted_category = class_names[predicted_index]
    confidence = probabilities[0][predicted_index].item()
    
    # Ensure category folder exists
    category_path = os.path.join(DATASET_PATH, predicted_category)
    os.makedirs(category_path, exist_ok=True)
    
    # Save image in the correct category folder
    image_save_path = os.path.join(category_path, filename)
    image.save(image_save_path)
    
    return predicted_category, confidence, image_save_path

# Streamlit UI
st.title("📂 Smart Image Categorizer")
st.write("Upload your images and let AI categorize them instantly!")

uploaded_files = st.file_uploader("Upload images (single or multiple)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        category, confidence, saved_path = predict_and_save(image, uploaded_file.name)
        
        st.image(image, caption=f"{uploaded_file.name} → {category} ({confidence:.2%})", use_column_width=True)
        st.success(f"✅ Categorized as: **{category}** (Confidence: {confidence:.2%})")
        st.info(f"📂 Image saved to: {saved_path}")
