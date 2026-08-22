"""
Loads the pneumonia X-ray classifier and runs inference on an uploaded image.
Deliberately separate from prediction_service.py since this is a different
modality (image, not tabular) with its own preprocessing pipeline.
"""

import json
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from app.core.config import settings

XRAY_MODEL_DIR = settings.ML_MODELS_DIR / "xray"
IMG_SIZE = 128
DEVICE = torch.device("cpu")

_model = None
_classes = None

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_xray_model():
    global _model, _classes

    manifest_path = XRAY_MODEL_DIR / "production_manifest.json"
    model_path = XRAY_MODEL_DIR / "production_model.pt"

    if not model_path.exists():
        print(f"[xray_service] WARNING: no xray model found at {model_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    _classes = manifest["classes"]  # ['NORMAL', 'PNEUMONIA']

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(_classes))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    _model = model
    print(f"[xray_service] Loaded xray model, classes={_classes}")


def is_ready() -> bool:
    return _model is not None


def is_likely_xray(image: Image.Image) -> bool:
    """
    Heuristic gate: chest X-rays are near-grayscale in virtually every pixel.
    Real photos/screenshots/UIs have SOME meaningfully colored pixels even
    if the overall average saturation looks low (e.g. white background with
    a few colored buttons/icons). We check what fraction of pixels have
    real R/G/B channel divergence, rather than averaging over the whole image.
    """
    arr = np.array(image).astype(int)  # shape (H, W, 3)
    channel_range = arr.max(axis=2) - arr.min(axis=2)  # per-pixel max-min across R,G,B
    colored_pixel_fraction = (channel_range > 20).mean()
    return colored_pixel_fraction < 0.02


def predict_xray(image_bytes: bytes) -> dict:
    if _model is None:
        raise RuntimeError("X-ray model is not loaded")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    if not is_likely_xray(image):
        return {
            "risk_label": "Invalid Input",
            "probability": 0.0,
            "model_version": "pneumonia_resnet18_v1",
            "valid_input": False,
        }

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = _model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pneumonia_idx = _classes.index("PNEUMONIA")
    pneumonia_prob = float(probs[pneumonia_idx])
    risk_label = "High Risk" if pneumonia_prob >= 0.5 else "Low Risk"

    return {
        "risk_label": risk_label,
        "probability": pneumonia_prob,
        "model_version": "pneumonia_resnet18_v1",
        "valid_input": True,
    }
