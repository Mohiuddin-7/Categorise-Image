import os
import json
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Load domain configuration
config_path = "domain_config.json"
with open(config_path, "r") as f:
    domain_config = json.load(f)

# Extract category names from domain_config.json
config_categories = list(domain_config.keys())

# Path to categorized images folder
dataset_path = "categorized_images"

# Apply data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(p=0.5),   # Flip images randomly
    transforms.RandomRotation(degrees=15),   # Rotate images by up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust colors
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random crop
    transforms.ToTensor()
])

# Load dataset with augmentation
dataset = ImageFolder(root=dataset_path, transform=transform)

# Extract dataset categories
dataset_categories = dataset.classes

# Check for inconsistencies
print("\n✅ Dataset Classes from ImageFolder:", dataset_categories)
print("\n✅ Categories in domain_config.json:", config_categories)

if set(dataset_categories) != set(config_categories):
    print("\n⚠️ WARNING: Mismatch between dataset classes and domain_config.json!")

# Count images per category
print("\n📊 Image Count Per Category:")
for category, idx in dataset.class_to_idx.items():
    category_path = os.path.join(dataset_path, category)
    num_images = len(os.listdir(category_path)) if os.path.exists(category_path) else 0
    print(f"  - {category}: {num_images} images")

print("\n✅ All categories have images!")
