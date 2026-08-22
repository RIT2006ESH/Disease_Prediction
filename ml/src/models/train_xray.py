"""
Scoped CNN classifier: chest X-ray -> Normal / Pneumonia.
Deliberately minimal for time constraints: frozen ResNet18 backbone,
only the final layer is trained, small image size, capped dataset subset.
NOT segmentation — outputs a single label + confidence per image.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATA_DIR = Path(r"C:\Users\Asus\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "xray"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 2
MAX_TRAIN_PER_CLASS = 500  # cap for speed on CPU
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale; ResNet expects 3ch
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_capped_subset(dataset, max_per_class: int):
    """Limits each class to max_per_class images, for CPU-feasible training time."""
    class_counts = {}
    indices = []
    for idx, (_, label) in enumerate(dataset.samples):
        class_counts.setdefault(label, 0)
        if class_counts[label] < max_per_class:
            indices.append(idx)
            class_counts[label] += 1
    return Subset(dataset, indices)


def train():
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
    print(f"Classes: {train_dataset.classes}")  # ['NORMAL', 'PNEUMONIA']

    train_subset = get_capped_subset(train_dataset, MAX_TRAIN_PER_CLASS)
    print(f"Training on {len(train_subset)} images (capped for speed)")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading pretrained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers except the final classifier — this is what keeps
    # training fast on CPU: we're not updating millions of conv weights,
    # just the final linear layer for our 2-class problem.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {running_loss/len(train_loader):.4f} — {elapsed:.1f}s")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"\nTest accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save model + class mapping
    torch.save(model.state_dict(), MODELS_DIR / "production_model.pt")
    import json
    with open(MODELS_DIR / "production_manifest.json", "w") as f:
        json.dump({
            "disease": "pneumonia_xray",
            "architecture": "resnet18_frozen_backbone",
            "classes": train_dataset.classes,
            "test_accuracy": accuracy,
            "img_size": IMG_SIZE,
        }, f, indent=2)

    print(f"Saved model to {MODELS_DIR / 'production_model.pt'}")


if __name__ == "__main__":
    train()