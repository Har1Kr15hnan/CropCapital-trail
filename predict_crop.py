import torch
import numpy as np
import xarray as xr
from torchvision import models, transforms
import torch.nn as nn
from pystac_client import Client
from odc.stac import load
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/crop_capital_cnn.pth'
CLASSES_PATH = 'models/classes.npy'
STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BUFFER_DEG = 0.003  # ~640m box

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD THE BRAIN ---
def load_model(num_classes):
    print(f"🧠 Loading model from {MODEL_PATH}...")
    # Use weights=None to avoid warnings and ensure clean load
    model = models.resnet18(weights=None) 
    
    # Recreate the 4-channel input layer (Must match training!)
    new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = new_conv1
    
    # Recreate the output layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the trained weights
    # weights_only=True is safer for security
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode
    return model

# --- 2. FETCH LIVE SATELLITE DATA ---
def fetch_satellite_image(lat, lon):
    print(f"🛰️  Fetching live view for ({lat}, {lon})...")
    client = Client.open(STAC_API_URL)
    bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]
    
    # Search for a clear image in 2023
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime="2023-01-01/2023-12-31",
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = search.item_collection()
    
    if len(items) == 0:
        print("❌ No cloud-free images found.")
        return None

    best_item = min(items, key=lambda item: item.properties['eo:cloud_cover'])
    
    # Download the 4 bands: Red, Green, Blue, NIR
    data = load(
        [best_item],
        bands=["red", "green", "blue", "nir"],
        bbox=bbox,
        resolution=10,
        chunks={}
    )
    
    # --- CRITICAL FIXES FOR SHAPE ERRORS ---
    
    # 1. Convert to numpy. Initial shape is often (Time, Band, Y, X) or (Band, Time, Y, X)
    img = data.to_array().values 
    
    # 2. Squeeze removes dimensions of size 1 (like Time=1)
    img = np.squeeze(img) 
    
    # 3. Ensure we have exactly (4, H, W)
    if img.ndim == 2: 
        # If we accidentally squeezed too much or only got 1 band
        # (This happens if only 1 band was downloaded, but we asked for 4)
        print("⚠️ Warning: Image has unusual shape, attempting fix...")
        img = np.stack([img]*4) 
        
    # Check if channels are last (H, W, 4) -> convert to (4, H, W)
    if img.shape[-1] == 4:
        img = img.transpose(2, 0, 1)
        
    # Final check: Must be 4 channels
    if img.shape[0] != 4:
        print(f"❌ Error: Expected 4 channels, got {img.shape[0]}")
        return None

    # Normalize (0-1 range)
    img = np.clip(img / 3000.0, 0, 1).astype(np.float32)
    
    # Convert to Tensor here for resizing
    tensor_img = torch.from_numpy(img)
    
    # 4. Force Resize to 64x64 (Matches Training Data)
    resize = transforms.Resize((64, 64), antialias=True)
    tensor_img = resize(tensor_img)
    
    return tensor_img

# --- 3. PREDICT ---
def predict(lat, lon):
    if not os.path.exists(CLASSES_PATH):
        print("❌ Error: classes.npy not found. Did you train the model?")
        return
    class_names = np.load(CLASSES_PATH)
    
    # Load Model
    model = load_model(len(class_names))
    
    # Get Image Tensor
    input_tensor = fetch_satellite_image(lat, lon)
    if input_tensor is None: return

    # Add Batch Dimension -> Shape becomes (1, 4, 64, 64)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Get Probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
        predicted_crop = class_names[predicted_idx.item()]
        conf_score = confidence.item() * 100

    print("\n" + "="*35)
    print(f"🌾 PREDICTION: {predicted_crop.upper()}")
    print(f"🎯 CONFIDENCE: {conf_score:.2f}%")
    print("="*35 + "\n")

if __name__ == "__main__":
    print("🌍 Crop Capital Prediction Engine")
    try:
        lat_in = float(input("Enter Latitude (e.g., 30.123): "))
        lon_in = float(input("Enter Longitude (e.g., 75.456): "))
        predict(lat_in, lon_in)
    except ValueError:
        print("❌ Invalid coordinates.")