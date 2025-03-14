import os
import json
from collections import Counter
from torchvision.datasets import ImageFolder

# Paths
dataset_path = "categorized_images"
domain_config_path = "domain_config.json"

# Load dataset using ImageFolder
dataset = ImageFolder(root=dataset_path)

# Count images in each class
category_counts = Counter()
for class_idx in dataset.targets:
    category_counts[dataset.classes[class_idx]] += 1

# Load domain_config.json
with open(domain_config_path, "r") as f:
    domain_config = json.load(f)

# Print dataset classes and domain config keys
print("\n✅ Dataset Classes from ImageFolder:", dataset.classes)
print("\n✅ Categories in domain_config.json:", list(domain_config.keys()))

# Check if classes match
if set(dataset.classes) == set(domain_config.keys()):
    print("\n✅ Class labels MATCH between dataset and domain_config.json!")
else:
    print("\n⚠️ WARNING: Mismatch between dataset classes and domain_config.json!")

# Print category counts
print("\n📊 Image Count Per Category:")
for category, count in category_counts.items():
    print(f"  - {category}: {count} images")

# Check for empty categories
empty_categories = [c for c in dataset.classes if category_counts[c] == 0]
if empty_categories:
    print("\n⚠️ WARNING: Some categories have 0 images:", empty_categories)
else:
    print("\n✅ All categories have images!")