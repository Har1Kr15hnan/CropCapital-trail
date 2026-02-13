"""
Training Script for Crop Classification Model using EuroSAT Dataset
Downloads and trains on real satellite imagery
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
EUROSAT_PATH = "training_data/eurosat/2750"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Map EuroSAT classes to our crop classes
EUROSAT_TO_CROP_MAP = {
    'AnnualCrop': 1,        # Wheat
    'PermanentCrop': 0,     # Paddy/Rice
    'Pasture': 6,           # Pulses
    'HerbaceousVegetation': 7,  # Vegetables
    'Forest': 8,            # Not crop (Fallow)
    'Highway': 8,           # Not crop
    'Industrial': 8,        # Not crop
    'Residential': 8,       # Not crop
    'River': 8,             # Not crop
    'SeaLake': 8            # Not crop
}

CROP_LABELS = {
    0: "Paddy/Rice",
    1: "Wheat",
    2: "Cotton",
    3: "Sugarcane",
    4: "Maize/Corn",
    5: "Soybean",
    6: "Pulses",
    7: "Vegetables",
    8: "Barren/Fallow"
}

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def calculate_ndvi(red, nir):
    """Normalized Difference Vegetation Index"""
    return (nir - red) / (nir + red + 1e-8)

def calculate_evi(red, nir, blue):
    """Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

def calculate_savi(red, nir, L=0.5):
    """Soil Adjusted Vegetation Index"""
    return ((nir - red) / (nir + red + L)) * (1 + L)

def calculate_gci(nir, green):
    """Green Chlorophyll Index"""
    return (nir / (green + 1e-8)) - 1

def extract_features_from_image(img_path):
    """
    Extract spectral features from multispectral satellite image
    EuroSAT format: 13 bands, we use first 4 (B, G, R, NIR)
    """
    try:
        # For EuroSAT TIFF files, we'd use rasterio
        # For now, simulate with OpenCV (converts to RGB)
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Extract bands (simulated)
        # In real EuroSAT: Band 4 = Red, Band 8 = NIR, Band 3 = Green, Band 2 = Blue
        # Here we approximate: R=red, G=green, B=blue, NIR=green*1.5 (approximation)
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        nir = green * 1.5  # Approximate NIR from green channel
        
        # Calculate indices
        ndvi = calculate_ndvi(red, nir)
        evi = calculate_evi(red, nir, blue)
        savi = calculate_savi(red, nir)
        gci = calculate_gci(nir, green)
        
        # Compile features
        features = {
            'ndvi_mean': np.mean(ndvi),
            'ndvi_std': np.std(ndvi),
            'ndvi_max': np.max(ndvi),
            'ndvi_min': np.min(ndvi),
            'evi_mean': np.mean(evi),
            'savi_mean': np.mean(savi),
            'gci_mean': np.mean(gci),
            'red_mean': np.mean(red),
            'nir_mean': np.mean(nir),
            'green_mean': np.mean(green),
            'blue_mean': np.mean(blue),
            'nir_red_ratio': np.mean(nir / (red + 1e-8)),
            'green_red_ratio': np.mean(green / (red + 1e-8)),
            'red_variance': np.var(red),
            'nir_variance': np.var(nir),
            'vegetation_coverage': np.sum(ndvi > 0.4) / ndvi.size * 100,
            'high_vigor_pct': np.sum(ndvi > 0.6) / ndvi.size * 100,
            'bare_soil_pct': np.sum(ndvi < 0.2) / ndvi.size * 100,
        }
        
        return features
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# ==========================================
# DATA LOADING
# ==========================================
def load_eurosat_dataset(dataset_path, max_samples_per_class=1000):
    """
    Load EuroSAT dataset and extract features
    """
    print("[Loading] Scanning EuroSAT dataset...")
    
    X = []
    y = []
    
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset path not found: {dataset_path}")
        print("[Info] Please download EuroSAT dataset:")
        print("       wget http://madm.dfki.de/files/sentinel/EuroSAT.zip")
        print("       unzip EuroSAT.zip -d training_data/eurosat/")
        return None, None
    
    # Iterate through class directories
    class_dirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    for class_name in class_dirs:
        if class_name not in EUROSAT_TO_CROP_MAP:
            continue
        
        crop_label = EUROSAT_TO_CROP_MAP[class_name]
        class_path = os.path.join(dataset_path, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.endswith(('.jpg', '.png', '.tif', '.tiff'))]
        
        # Limit samples per class for balanced dataset
        image_files = image_files[:max_samples_per_class]
        
        print(f"[Loading] {class_name} ({len(image_files)} samples) -> {CROP_LABELS[crop_label]}")
        
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_file)
            features = extract_features_from_image(img_path)
            
            if features is not None:
                X.append(features)
                y.append(crop_label)
    
    print(f"\n[Loaded] Total samples: {len(X)}")
    return pd.DataFrame(X), np.array(y)

# ==========================================
# ALTERNATIVE: KAGGLE DATASET LOADER
# ==========================================
def load_kaggle_agriculture_dataset(dataset_path="training_data/crops"):
    """
    Load Kaggle agricultural crops dataset
    Expected structure: crops/rice/, crops/wheat/, etc.
    """
    print("[Loading] Scanning Kaggle Agriculture dataset...")
    
    X = []
    y = []
    
    crop_name_map = {
        'rice': 0, 'paddy': 0,
        'wheat': 1,
        'cotton': 2,
        'sugarcane': 3, 'sugar': 3,
        'maize': 4, 'corn': 4,
        'soybean': 5,
        'pulses': 6, 'lentils': 6, 'chickpea': 6,
        'vegetables': 7, 'tomato': 7, 'potato': 7,
        'jute': 2,  # Map to cotton (similar)
    }
    
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset path not found: {dataset_path}")
        return None, None
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Find matching crop label
        crop_label = None
        for key, value in crop_name_map.items():
            if key in class_name.lower():
                crop_label = value
                break
        
        if crop_label is None:
            print(f"[Skip] Unknown crop: {class_name}")
            continue
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"[Loading] {class_name} ({len(image_files)} samples) -> {CROP_LABELS[crop_label]}")
        
        for img_file in tqdm(image_files[:1000], desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_file)
            features = extract_features_from_image(img_path)
            
            if features is not None:
                X.append(features)
                y.append(crop_label)
    
    print(f"\n[Loaded] Total samples: {len(X)}")
    return pd.DataFrame(X), np.array(y)

# ==========================================
# MODEL TRAINING
# ==========================================
def train_model(X, y):
    """
    Train Random Forest Classifier
    """
    print("\n" + "="*60)
    print("TRAINING CROP CLASSIFICATION MODEL")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[Split] Training: {len(X_train)} | Testing: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n[Training] Fitting Random Forest (200 trees)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n[Evaluation] Testing on holdout set...")
    y_pred = model.predict(X_test_scaled)
    
    # Accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"\n✓ Test Accuracy: {accuracy:.2f}%")
    
    # Cross-validation
    print("\n[Cross-Validation] 5-fold CV...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.2f}% (+/- {cv_scores.std():.2f}%)")
    
    # Feature importance
    feature_names = list(X.columns)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\n[Features] Top 10 Important Features:")
    for i in range(min(10, len(sorted_idx))):
        print(f"  {i+1}. {feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")
    
    # Classification report
    print("\n[Report] Per-Class Performance:")
    unique_labels = np.unique(y_test)
    target_names = [CROP_LABELS[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, unique_labels)
    
    return model, scaler, accuracy

# ==========================================
# VISUALIZATION
# ==========================================
def plot_confusion_matrix(cm, labels):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    label_names = [CROP_LABELS[i] for i in labels]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    
    plt.title('Crop Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    print(f"\n[Saved] Confusion matrix: {MODEL_DIR}/confusion_matrix.png")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("="*60)
    print("   CROP CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Choose dataset source
    print("\n[Options] Available datasets:")
    print("  1. EuroSAT (Satellite imagery, European focus)")
    print("  2. Kaggle Agriculture (High-res images, Indian crops)")
    print("  3. Generate synthetic data (Testing only)")
    
    choice = input("\nSelect dataset (1/2/3): ").strip()
    
    X, y = None, None
    
    if choice == '1':
        X, y = load_eurosat_dataset(EUROSAT_PATH)
    elif choice == '2':
        X, y = load_kaggle_agriculture_dataset()
    elif choice == '3':
        print("[Synthetic] Generating 5000 training samples...")
        from crop_ai_engine_v3 import generate_training_data
        X, y = generate_training_data(n_samples=5000)
    else:
        print("[Error] Invalid choice")
        return
    
    if X is None or len(X) == 0:
        print("[Error] No data loaded. Exiting.")
        return
    
    # Train model
    model, scaler, accuracy = train_model(X, y)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "crop_classifier.pkl")
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n[Saved] Model: {model_path}")
    print(f"[Saved] Scaler: {scaler_path}")
    
    # Save metadata
    metadata = {
        'accuracy': accuracy,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'crop_labels': CROP_LABELS
    }
    
    import json
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel Accuracy: {accuracy:.2f}%")
    print("You can now use the model with: python crop_ai_engine_v3.py")

if __name__ == "__main__":
    main()
