import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import os

with open("domain_config.json", "r") as f:
    domain_config = json.load(f)
    class_names = list(domain_config.keys())

num_classes = len(class_names)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(1280, num_classes)
model.load_state_dict(torch.load("custom_image_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("📂 AI-Powered Image Categorization")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Categorize Image"):
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()

        predicted_category = class_names[predicted_index]
        confidence = probabilities[0][predicted_index].item()
        st.success(f"✅ **Predicted Category:** {predicted_category} ({confidence:.2%} confidence)")
