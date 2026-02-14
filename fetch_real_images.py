import pandas as pd
import xarray as xr
from pystac_client import Client
from odc.stac import load
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_CSV = 'training_data/target_locations.csv'
OUTPUT_DIR = 'training_data/images'
# We use the free AWS Earth Search API (No API Key needed for Sentinel-2)
STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Box size: 640m x 640m (approx 64 pixels at 10m resolution)
# This is standard for many crop models (EuroSAT uses 64x64)
BUFFER_DEG = 0.003  

def fetch_images():
    # 1. Load your targets
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Did you run the previous step?")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"🚀 Loaded {len(df)} targets. Starting satellite download...")

    # 2. Connect to the Archive
    client = Client.open(STAC_API_URL)

    # 3. Loop through every farm
    for index, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        crop = row['crop']
        
        # Create folder: training_data/images/wheat/
        save_folder = os.path.join(OUTPUT_DIR, crop)
        os.makedirs(save_folder, exist_ok=True)
        
        filename = f"{crop}_{index}_{lat:.4f}_{lon:.4f}.nc"
        filepath = os.path.join(save_folder, filename)
        
        if os.path.exists(filepath):
            print(f"⏭️  Skipping {filename} (Already exists)")
            continue

        print(f"🛰️  [{index+1}/{len(df)}] Searching clean image for {crop} at {lat:.3f}, {lon:.3f}...")

        try:
            # 4. Define the Search Box (Area of Interest)
            bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]

            # 5. Find the best image from the last year (2023-2024)
            # We filter for <10% cloud cover to ensure clean data
            search = client.search(
                collections=[COLLECTION],
                bbox=bbox,
                datetime="2023-01-01/2023-12-31",
                query={"eo:cloud_cover": {"lt": 10}}
            )
            
            items = search.item_collection()
            
            if len(items) == 0:
                print(f"   ⚠️ No cloud-free images found for this spot.")
                continue

            # Take the least cloudy image
            best_item = min(items, key=lambda item: item.properties['eo:cloud_cover'])

            # 6. Download ONLY the pixels we need (The "Chip")
            # We load bands: Red (B04), Green (B03), Blue (B02), NIR (B08)
            # This uses 'odc-stac' to stream data without downloading the whole gigabyte file
            data = load(
                [best_item],
                bands=["red", "green", "blue", "nir"],
                bbox=bbox,
                resolution=10, # 10 meters per pixel
                chunks={}      # Load into memory
            )

            # 7. Save as NetCDF (Data Cube) or TIF
            # Squeeze removes time dimension (we just want the 2D image)
            img_slice = data.isel(time=0).to_array() 
            
            # Convert to a standard format (C, H, W)
            # You can save as .npy (numpy) or .nc (xarray)
            # Here we save as .nc which preserves band names
            data.to_netcdf(filepath)
            
            print(f"   ✅ Saved {filename}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n🏁 Download Complete!")

if __name__ == "__main__":
    fetch_images()