#!/bin/bash

# ===========================================
# CropCapital AI - Quick Setup Script
# ===========================================

echo "================================================"
echo "   CropCapital AI Setup & Dataset Downloader"
echo "================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Install Dependencies
echo -e "\n${BLUE}[1/4] Installing Python dependencies...${NC}"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Step 2: Create Directory Structure
echo -e "\n${BLUE}[2/4] Creating directory structure...${NC}"
mkdir -p training_data/eurosat
mkdir -p training_data/crops
mkdir -p models
mkdir -p outputs

echo -e "${GREEN}✓ Directories created${NC}"

# Step 3: Ask user which dataset to download
echo -e "\n${BLUE}[3/4] Dataset Selection${NC}"
echo "Which dataset would you like to download?"
echo "  1) EuroSAT (2.8 GB) - Best for production [RECOMMENDED]"
echo "  2) Kaggle Agriculture (600 MB) - Requires Kaggle API setup"
echo "  3) Skip download (use synthetic data for testing)"
echo ""
read -p "Enter choice (1/2/3): " dataset_choice

case $dataset_choice in
    1)
        echo -e "\n${BLUE}Downloading EuroSAT dataset...${NC}"
        cd training_data/eurosat
        
        # Check if already downloaded
        if [ -f "EuroSAT.zip" ]; then
            echo "EuroSAT.zip already exists. Skipping download."
        else
            echo "Downloading from server (2.8 GB)..."
            wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
            
            if [ $? -ne 0 ]; then
                echo "Download failed. Please check your internet connection."
                exit 1
            fi
        fi
        
        # Extract
        if [ ! -d "2750" ]; then
            echo "Extracting dataset..."
            unzip -q EuroSAT.zip
            echo -e "${GREEN}✓ EuroSAT dataset ready${NC}"
        else
            echo "Dataset already extracted."
        fi
        
        cd ../..
        ;;
    
    2)
        echo -e "\n${BLUE}Kaggle Dataset Download${NC}"
        echo "Prerequisites:"
        echo "  1. Install Kaggle CLI: pip install kaggle"
        echo "  2. Get API credentials from kaggle.com/settings"
        echo "  3. Place kaggle.json in ~/.kaggle/"
        echo ""
        read -p "Have you completed setup? (y/n): " kaggle_ready
        
        if [ "$kaggle_ready" == "y" ]; then
            # Check if kaggle CLI exists
            if command -v kaggle &> /dev/null; then
                echo "Downloading Kaggle Agricultural dataset..."
                kaggle datasets download -d mdwaquarazam/agricultural-crops-image-classification -p training_data/
                
                cd training_data
                unzip -q agricultural-crops-image-classification.zip -d crops/
                cd ..
                
                echo -e "${GREEN}✓ Kaggle dataset ready${NC}"
            else
                echo "Kaggle CLI not found. Install with: pip install kaggle"
                exit 1
            fi
        else
            echo "Skipping Kaggle download. You can do this manually later."
        fi
        ;;
    
    3)
        echo -e "${GREEN}✓ Skipping download. Will use synthetic data.${NC}"
        ;;
    
    *)
        echo "Invalid choice. Using synthetic data."
        ;;
esac

# Step 4: Train Model
echo -e "\n${BLUE}[4/4] Training Model${NC}"
read -p "Would you like to train the model now? (y/n): " train_now

if [ "$train_now" == "y" ]; then
    echo "Starting training..."
    python train_crop_model.py
else
    echo "Skipping training. You can train later with:"
    echo "  python train_crop_model.py"
fi

# Final Instructions
echo ""
echo "================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Next Steps:"
echo "  1. Start API server: python crop_ai_engine_v3.py"
echo "  2. Test endpoint:"
echo "     curl -X POST http://localhost:5000/analyze-farm \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"lat\": 28.6139, \"lon\": 77.2090, \"acres\": 5}'"
echo ""
echo "For Sentinel Hub API (optional):"
echo "  export SENTINEL_HUB_CLIENT_ID='your_id'"
echo "  export SENTINEL_HUB_CLIENT_SECRET='your_secret'"
echo ""
echo "Documentation: See TRAINING_GUIDE.md"
echo "================================================"
