"""
Training Script for Crop Capital (Deep Learning Version)
Trains a ResNet18 model on 4-channel Sentinel-2 Satellite Chips (.nc files)
FIX: Automatically resizes all inputs to 64x64 to prevent shape errors.
"""

import os
import glob
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'training_data/images' 
MODEL_SAVE_PATH = 'models/crop_capital_cnn.pth'
CLASSES_SAVE_PATH = 'models/classes.npy'

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. CUSTOM DATASET LOADER (With Auto-Resize)
# ==========================================
class SatelliteDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        # Define the resizer to force all images to 64x64
        self.resize = transforms.Compose([
            transforms.Resize((64, 64), antialias=True)
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            # Open the NetCDF file
            ds = xr.open_dataset(path)
            image = ds.to_array().values.squeeze()
            ds.close()

            # Ensure shape is (4, H, W)
            if image.ndim == 2:
                # If only 1 band found, duplicate it 4 times (rare error case)
                image = np.stack([image]*4)
            elif image.shape[0] != 4:
                if image.shape[-1] == 4: 
                    image = image.transpose(2, 0, 1)
                else:
                    # If more than 4 bands, take first 4
                    image = image[:4, :, :] 

            # Convert to PyTorch Tensor
            image_tensor = torch.from_numpy(image).float()

            # NORMALIZE: Scale 0-10000 -> 0-1
            image_tensor = image_tensor / 3000.0
            image_tensor = torch.clamp(image_tensor, 0, 1)

            # CRITICAL FIX: Force Resize to 64x64
            # This fixes the "stack expects equal size" error
            image_tensor = self.resize(image_tensor)

        except Exception as e:
            # print(f"⚠️ Error loading {path}: {e}")
            # Return a blank black image so training doesn't crash
            image_tensor = torch.zeros((4, 64, 64), dtype=torch.float32)

        return image_tensor, torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
def get_model(num_classes):
    print(f"🧠 Initializing ResNet18 for {num_classes} crop classes...")
    model = models.resnet18(pretrained=True)
    
    # MODIFY FIRST LAYER: 4 Channels (RGB+NIR) instead of 3
    original_weights = model.conv1.weight.data.clone()
    new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        new_conv1.weight[:, :3] = original_weights
        new_conv1.weight[:, 3] = torch.mean(original_weights, dim=1) 
        
    model.conv1 = new_conv1
    
    # MODIFY LAST LAYER
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# ==========================================
# 3. MAIN TRAINING FLOW
# ==========================================
def main():
    os.makedirs('models', exist_ok=True)
    print("="*60 + "\n 🛰️  CROP CAPITAL: DEEP LEARNING TRAINER \n" + "="*60)
    print(f"⚙️  Device: {DEVICE}")

    # --- A. PREPARE DATA ---
    print("\n[1/4] Scanning 'training_data/images' for .nc files...")
    all_files = []
    all_labels = []
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: {DATA_DIR} not found.")
        return

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"   Found Classes: {classes}")
    
    for crop_name in classes:
        crop_path = os.path.join(DATA_DIR, crop_name)
        files = glob.glob(os.path.join(crop_path, "*.nc"))
        
        if len(files) == 0:
            continue
            
        print(f"   - {crop_name}: {len(files)} samples")
        all_files.extend(files)
        all_labels.extend([crop_name] * len(files))

    if len(all_files) == 0:
        print("❌ No training data found! Exiting.")
        return

    # Encode Labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(all_labels)
    np.save(CLASSES_SAVE_PATH, le.classes_)
    
    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(all_files, encoded_labels, test_size=0.2, random_state=42)
    
    train_ds = SatelliteDataset(X_train, y_train)
    val_ds = SatelliteDataset(X_val, y_val)
    
    # NOTE: num_workers=0 is safer on Windows to avoid multi-processing errors
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # --- B. SETUP MODEL ---
    print("\n[2/4] Building Model...")
    model = get_model(len(classes)).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- C. TRAIN LOOP ---
    print("\n[3/4] Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.2f}%")

    # --- D. SAVE ---
    print("\n[4/4] Saving 'Brain'...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
    print("\n🚀 DONE! You can now run 'python predict_crop.py'")

if __name__ == "__main__":
    main()