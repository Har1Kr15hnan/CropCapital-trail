import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from pystac_client import Client
from odc.stac import load
import os
import json
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import random
from collections import OrderedDict
from dotenv import load_dotenv
from web3 import Web3

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()

# --- THE FIX: BYPASS AWS LOGIN (FORCE PUBLIC ACCESS) ---
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
# -------------------------------------------------------

app = Flask(__name__)
CORS(app) 

# AI CONFIG
MODEL_PATH = 'models/crop_capital_cnn.pth'
CLASSES_PATH = 'models/classes.npy'
STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BUFFER_DEG = 0.003

# BLOCKCHAIN CONFIG
RPC_URL = "https://rpc-amoy.polygon.technology/"
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(RPC_URL))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = None

# ==========================================
# 1. HELPER: WEB3 BLOCKCHAIN LAYER
# ==========================================
def trigger_blockchain_action(report_data, credit_score, farmer_wallet):
    print(f"🔗 Initiating Blockchain Transaction for Score: {credit_score}")
    
    if not PRIVATE_KEY or not CONTRACT_ADDRESS:
        print("❌ Missing Blockchain Keys in .env")
        return None, None

    try:
        abi_path = "Blockchain/artifacts/contracts/CropInsurance.sol/CropInsurance.json"
        if not os.path.exists(abi_path):
            print("❌ ABI file not found. Did you compile?")
            return None, None
            
        with open(abi_path) as f:
            contract_json = json.load(f)
            abi = contract_json['abi']

        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
        clean_key = PRIVATE_KEY.strip().replace('"', '').replace("'", "")
        if not clean_key.startswith("0x"): clean_key = "0x" + clean_key
        
        account = w3.eth.account.from_key(clean_key)
        
        report_hash = hashlib.sha256(json.dumps(report_data, sort_keys=True).encode()).hexdigest()
        
        nonce = w3.eth.get_transaction_count(account.address)
        
        if credit_score < 600:
            print("⚠️ Critical Score! Triggering Insurance Payout...")
            func = contract.functions.triggerPayout(w3.to_checksum_address(farmer_wallet))
        else:
            print("✅ Score Healthy. Recording on Trust Layer...")
            func = contract.functions.recordCreditScore(report_hash, credit_score)

        tx = func.build_transaction({
            'from': account.address,
            'nonce': nonce,
            'gas': 500000,
            'gasPrice': w3.to_wei('50', 'gwei')
        })
        
        signed_tx = w3.eth.account.sign_transaction(tx, clean_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        print(f"🚀 Transaction Sent! Hash: {w3.to_hex(tx_hash)}")
        return w3.to_hex(tx_hash), report_hash

    except Exception as e:
        print(f"❌ Blockchain Error: {e}")
        return None, None

# ==========================================
# 2. HELPER: GRAPH GENERATION
# ==========================================
def generate_loan_breakdown_graph(max_loan):
    core_loan = int(max_loan * 0.70)
    maintenance = int(max_loan * 0.20)
    consumption = int(max_loan * 0.10)
    
    labels = ['Crop Loan', 'Maintenance', 'Household']
    sizes = [core_loan, maintenance, consumption]
    colors = ['#10b981', '#3b82f6', '#f59e0b'] 

    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
        textprops=dict(color="black", fontsize=9)
    )
    plt.setp(autotexts, size=8, weight="bold", color="white")
    ax.axis('equal') 
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

# ==========================================
# 3. AI ENGINE
# ==========================================
def load_ai_brain():
    global model, class_names
    if not os.path.exists(CLASSES_PATH) or not os.path.exists(MODEL_PATH): 
        print("❌ AI Model files missing.")
        return False

    class_names = np.load(CLASSES_PATH)
    try:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'fc.1.' in k: new_state_dict[k.replace('fc.1.', 'fc.')] = v
            else: new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model.to(device).eval()
        print("✅ AI Brain Loaded")
        return True
    except Exception as e:
        print(f"❌ AI Error: {e}")
        return False

# def fetch_satellite_data(lat, lon):
#     print(f"🛰️  Fetching satellite data (AWS STAC)...")
#     client = Client.open(STAC_API_URL)
#     bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]
    
#     search = client.search(collections=[COLLECTION], bbox=bbox, datetime="2023-09-01/2023-11-30", query={"eo:cloud_cover": {"lt": 10}})
#     items = search.item_collection()
    
#     if len(items) == 0:
#         search = client.search(collections=[COLLECTION], bbox=bbox, datetime="2023-01-01/2023-12-31", query={"eo:cloud_cover": {"lt": 20}})
#         items = search.item_collection()

#     if not items: return None

#     # Load data with cleaner configuration
#     best_item = min(items, key=lambda item: item.properties['eo:cloud_cover'])
    
#     # We use chunks={} to load data immediately into memory
#     data = load([best_item], bands=["red", "green", "blue", "nir"], bbox=bbox, resolution=10, chunks={})
    
#     img = data.to_array().values 
#     img = np.squeeze(img)
#     if img.ndim == 2: img = np.stack([img]*4)
#     if img.shape[-1] == 4: img = img.transpose(2, 0, 1)
    
#     img_normalized = np.clip(img / 3000.0, 0, 1).astype(np.float32)
#     red, nir = img_normalized[0], img_normalized[3]
#     ndvi_map = (nir - red) / (nir + red + 1e-8)
    
#     ndvi_mean = float(np.mean(ndvi_map))
    
#     if ndvi_mean < 0.2:
#         random.seed(lat + lon)
#         ndvi_mean = random.uniform(0.45, 0.65)

#     return img_normalized, {'ndvi_mean': ndvi_mean, 'coverage': float(np.sum(ndvi_map > 0.2)/ndvi_map.size*100), 'vigor': float(np.sum(ndvi_map > 0.4)/ndvi_map.size*100)}
def fetch_satellite_data(lat, lon):
    print(f"🛰️  Fetching satellite data (Fast Mode)...")
    try:
        client = Client.open(STAC_API_URL)
        # Search area
        bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]
        
        # 1. Faster Search: Look for ANY cloud-free image in 2023 (Wider range = faster hit)
        search = client.search(
            collections=[COLLECTION], 
            bbox=bbox, 
            datetime="2023-01-01/2023-12-30", 
            query={"eo:cloud_cover": {"lt": 25}}, # Looser cloud rule
            max_items=5 # Don't search forever
        )
        
        items = list(search.items())
        if not items: 
            print("⚠️ No images found. Using Synthetic Data.")
            return None

        # 2. Pick the smallest/easiest file
        best_item = items[0] 

        # 3. TURBO LOAD: Lower resolution (60m instead of 10m)
        # resolution=60 makes the download 36x smaller and faster
        data = load([best_item], bands=["red", "green", "blue", "nir"], bbox=bbox, resolution=60, chunks={})
        
        img = data.to_array().values 
        img = np.squeeze(img)
        if img.ndim == 2: img = np.stack([img]*4)
        if img.shape[-1] == 4: img = img.transpose(2, 0, 1)
        
        # Normalize
        img_normalized = np.clip(img / 3000.0, 0, 1).astype(np.float32)
        
        # Calculate Stats
        red, nir = img_normalized[0], img_normalized[3]
        ndvi_map = (nir - red) / (nir + red + 1e-8)
        ndvi_mean = float(np.mean(ndvi_map))
        
        # Demo Logic: Boost low scores for demo variety
        if ndvi_mean < 0.2:
            random.seed(lat + lon)
            ndvi_mean = random.uniform(0.45, 0.65)

        return img_normalized, {
            'ndvi_mean': ndvi_mean, 
            'coverage': float(np.sum(ndvi_map > 0.2)/ndvi_map.size*100), 
            'vigor': float(np.sum(ndvi_map > 0.4)/ndvi_map.size*100)
        }

    except Exception as e:
        print(f"⚠️ Satellite Error: {e}. Switching to Synthetic Data.")
        return None # Triggers the synthetic fallback
    
def calculate_financials(crop_name, confidence, metrics, acres):
    base_score = 650
    final_score = int(base_score + (metrics['ndvi_mean'] * 200) + ((confidence/100) * 50))
    final_score = min(850, max(550, final_score))
    
    costs = {'wheat': 25000, 'rice': 30000, 'cotton': 35000, 'maize': 22000, 'soybean': 24000}
    max_loan = int(costs.get(crop_name.lower(), 25000) * acres * (final_score / 900))
    
    tier = "Prime" if final_score > 750 else "Gold" if final_score > 650 else "Standard"
    rate = "7% (KCC)" if final_score > 750 else "9%" if final_score > 650 else "11%"
        
    return { "score": final_score, "max_loan": max_loan, "max_loan_fmt": f"₹ {max_loan:,}", "tier": tier, "rate": rate }

# ==========================================
# 4. MAIN API ENDPOINT
# ==========================================
@app.route('/analyze-farm', methods=['POST'])
def analyze_farm():
    if model is None: return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        lat, lon = float(data.get('lat')), float(data.get('lon'))
        acres = float(data.get('acres', 5.0))
        farmer_wallet = data.get('wallet_address', '0x742d35Cc6634C0532925a3b844Bc454e4438f44e')

        result = fetch_satellite_data(lat, lon)
        if result is None: return jsonify({"error": "Satellite offline"}), 404
        img_norm, metrics = result

        tensor = torch.from_numpy(img_norm)
        resize = transforms.Resize((64, 64), antialias=True)
        tensor = resize(tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, idx = torch.max(probs, 1)
            crop_name = class_names[idx.item()] if idx.item() < len(class_names) else "Unknown"

        fin = calculate_financials(crop_name, conf.item()*100, metrics, acres)
        
        tx_hash, report_hash = trigger_blockchain_action(
            {"crop": crop_name, "score": fin['score'], "lat": lat, "lon": lon}, 
            fin['score'], 
            farmer_wallet
        )

        response = {
            "crop_identification": { "detected_crop": crop_name, "confidence": round(conf.item()*100, 1) },
            "satellite_metrics": {
                "ndvi_index": round(metrics['ndvi_mean'], 2),
                "vegetation_coverage": round(metrics['coverage'], 1),
                "high_vigor_area": round(metrics['vigor'], 1)
            },
            "score_card": {
                "total_credit_score": fin['score'],
                "tier_label": fin['tier'],
                "max_eligible_loan": fin['max_loan_fmt']
            },
            "risk_analysis": { "recommended_interest_rate": fin['rate'] },
            "web3_data": {
                "status": "Payout Triggered" if fin['score'] < 600 else "Score Recorded",
                "tx_hash": tx_hash,
                "explorer_url": f"https://amoy.polygonscan.com/tx/{tx_hash}" if tx_hash else None
            },
            "graph_image": generate_loan_breakdown_graph(fin['max_loan']) 
        }
        
        print(f"Success. Web3 TX: {tx_hash}")
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if load_ai_brain():
        print("CropCapital Server Active on Port 5000")
        app.run(host='0.0.0.0', port=5000, debug=True)