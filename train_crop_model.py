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
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'training_data/images'  # Update this to your actual data path
MODEL_SAVE_PATH = 'models/crop_capital_cnn.pth'
CLASSES_SAVE_PATH = 'models/classes.npy'

BATCH_SIZE = 32
EPOCHS = 30 # Increased to allow EarlyStopping room to work
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. UPDATED DATASET (With Augmentation)
# ==========================================
class SatelliteDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        # Added Random Flips to make the model invariant to satellite orientation
        self.transform = transforms.Compose([
            transforms.Resize((64, 64), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5) if augment else nn.Identity(),
            transforms.RandomVerticalFlip(p=0.5) if augment else nn.Identity(),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            ds = xr.open_dataset(path)
            image = ds.to_array().values.squeeze()
            ds.close()

            if image.ndim == 2: image = np.stack([image]*4)
            elif image.shape[0] != 4:
                image = image.transpose(2, 0, 1) if image.shape[-1] == 4 else image[:4, :, :] 

            image_tensor = torch.from_numpy(image).float() / 3000.0
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_tensor = self.transform(image_tensor)

        except Exception:
            image_tensor = torch.zeros((4, 64, 64), dtype=torch.float32)

        return image_tensor, torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 2. MODEL DEFINITION (With Dropout)
# ==========================================
def get_model(num_classes):
    print(f"🧠 Building ResNet18 + Dropout for {num_classes} classes...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # 4-Channel Input (RGB + NIR)
    original_weights = model.conv1.weight.data.clone()
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = original_weights
        model.conv1.weight[:, 3] = torch.mean(original_weights, dim=1) 
    
    # ADD DROPOUT: Injects a layer to prevent overfitting before the final decision
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# ==========================================
# 3. TRAINING LOGIC
# ==========================================
def main():
    os.makedirs('models', exist_ok=True)
    
    # Data Setup
    all_files, all_labels = [], []
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for crop_name in classes:
        files = glob.glob(os.path.join(DATA_DIR, crop_name, "*.nc"))
        all_files.extend(files)
        all_labels.extend([crop_name] * len(files))

    le = LabelEncoder()
    encoded_labels = le.fit_transform(all_labels)
    np.save(CLASSES_SAVE_PATH, le.classes_)
    
    X_train, X_val, y_train, y_val = train_test_split(all_files, encoded_labels, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(SatelliteDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SatelliteDataset(X_val, y_val, augment=False), batch_size=BATCH_SIZE)

    model = get_model(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # CALLBACK: Reduce LR when accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    history = {'loss': [], 'val_acc': []}
    best_acc = 0.0
    patience_counter = 0

    print("\n🚀 Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                _, predicted = torch.max(model(images).data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        history['loss'].append(train_loss/len(train_loader))
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} | Loss: {history['loss'][-1]:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early Stopping Logic
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(val_acc)
        if patience_counter >= 7: 
            print("🛑 Early stopping triggered!")
            break

if __name__ == "__main__":
    main()