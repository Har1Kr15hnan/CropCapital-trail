import torch
import numpy as np
import xarray as xr
from torchvision import models, transforms
import torch.nn as nn
from pystac_client import Client
from odc.stac import load
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for Next.js

MODEL_PATH = 'models/crop_capital_cnn.pth'
CLASSES_PATH = 'models/classes.npy'
STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BUFFER_DEG = 0.003  # ~640m box

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables to hold model in memory
model = None
class_names = None

# ==========================================
# 1. AI ENGINE (Model & Satellite)
# ==========================================

def load_ai_brain():
    global model, class_names
    if not os.path.exists(CLASSES_PATH):
        print("❌ Error: classes.npy not found.")
        return False
    
    print(f"🧠 Loading AI Model from {MODEL_PATH}...")
    class_names = np.load(CLASSES_PATH)
    
    # Initialize Architecture
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("✅ Model Loaded Successfully")
        return True
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")
        return False

def fetch_satellite_data(lat, lon):
    print(f"🛰️  Fetching live view for ({lat}, {lon})...")
    client = Client.open(STAC_API_URL)
    bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]
    
    # Search 2023-2024 for best cloud-free image
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime="2023-01-01/2023-12-31",
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = search.item_collection()
    
    if len(items) == 0: return None

    best_item = min(items, key=lambda item: item.properties['eo:cloud_cover'])
    
    data = load(
        [best_item],
        bands=["red", "green", "blue", "nir"],
        bbox=bbox,
        resolution=10,
        chunks={}
    )
    
    img = data.to_array().values 
    img = np.squeeze(img) 
    
    # Fix Shapes
    if img.ndim == 2: img = np.stack([img]*4)
    if img.shape[-1] == 4: img = img.transpose(2, 0, 1)
    
    # Normalize for AI
    img_normalized = np.clip(img / 3000.0, 0, 1).astype(np.float32)
    
    # Calculate Real NDVI Stats for the Frontend
    # Band 0=Red, Band 3=NIR
    red = img_normalized[0]
    nir = img_normalized[3]
    ndvi_map = (nir - red) / (nir + red + 1e-8)
    
    stats = {
        'ndvi_mean': float(np.mean(ndvi_map)),
        'coverage': float(np.sum(ndvi_map > 0.3) / ndvi_map.size * 100),
        'vigor': float(np.sum(ndvi_map > 0.6) / ndvi_map.size * 100)
    }

    return img_normalized, stats

# ==========================================
# 2. FINANCIAL LOGIC (The "Fintech" Layer)
# ==========================================
def calculate_financials(crop_name, confidence, metrics, acres):
    # Base logic: Healthier crop (NDVI) + Higher Confidence = Better Credit Score
    
    base_score = 650
    ndvi_bonus = metrics['ndvi_mean'] * 150 # Up to +150 points
    conf_bonus = (confidence / 100) * 50    # Up to +50 points
    
    final_score = int(base_score + ndvi_bonus + conf_bonus)
    final_score = min(900, max(300, final_score)) # Cap between 300-900
    
    # Loan logic based on crop type (Cost of cultivation per acre)
    crop_costs = {
        'wheat': 25000,
        'rice': 30000,
        'cotton': 35000,
        'sugarcane': 45000,
        'maize': 20000,
        'soybean': 22000
    }
    
    # Default to 20k if crop unknown
    cost_per_acre = crop_costs.get(crop_name.lower(), 20000)
    
    # Loan Eligibility = Cost * Acres * (Score Factor)
    score_factor = final_score / 900
    max_loan = int(cost_per_acre * acres * score_factor)
    
    tier = "Standard"
    rate = "12%"
    
    if final_score > 750:
        tier = "Prime: Instant Approval"
        rate = "7% (Subsidized)"
    elif final_score > 650:
        tier = "Gold: Fast Track"
        rate = "9%"
        
    return {
        "score": final_score,
        "max_loan": f"₹ {max_loan:,}",
        "tier": tier,
        "rate": rate
    }

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.route('/analyze-farm', methods=['POST'])
def analyze_farm():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        req_data = request.json
        lat = float(req_data.get('lat'))
        lon = float(req_data.get('lon'))
        acres = float(req_data.get('acres', 5.0))

        # 1. Get Satellite Data
        result = fetch_satellite_data(lat, lon)
        if result is None:
            return jsonify({"error": "No clear satellite image found"}), 404
            
        img_norm, metrics = result

        # 2. Prepare for AI
        tensor_img = torch.from_numpy(img_norm)
        resize = transforms.Resize((64, 64), antialias=True)
        tensor_img = resize(tensor_img).unsqueeze(0).to(device)

        # 3. Run Prediction
        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
            crop_name = class_names[predicted_idx.item()]
            conf_val = confidence.item() * 100

        # 4. Calculate Financials
        finances = calculate_financials(crop_name, conf_val, metrics, acres)

        # 5. Construct JSON Response (Matching Frontend Structure)
        response = {
            "crop_identification": {
                "detected_crop": crop_name,
                "confidence": round(conf_val, 1)
            },
            "satellite_metrics": {
                "ndvi_index": round(metrics['ndvi_mean'], 2),
                "vegetation_coverage": round(metrics['coverage'], 1),
                "high_vigor_area": round(metrics['vigor'], 1)
            },
            "score_card": {
                "total_credit_score": finances['score'],
                "tier_label": finances['tier'],
                "max_eligible_loan": finances['max_loan']
            },
            "risk_analysis": {
                "recommended_interest_rate": finances['rate']
            },
            # Optional: Send a static graph placeholder or generate one
            "graph_image": None 
        }

        print(f"✅ Analysis Complete: {crop_name} ({conf_val:.1f}%) - Score: {finances['score']}")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 4. RUN SERVER
# ==========================================
if __name__ == '__main__':
    # Load model once at startup
    if load_ai_brain():
        print("🚀 CropCapital Server Running on Port 5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to start server. Check model paths.")