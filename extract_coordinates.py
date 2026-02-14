import os
import glob
import xarray as xr
import pandas as pd

# --- CONFIGURATION ---
# Point to the folder where you EXTRACTED the files
# Use 'r' strings to handle Windows paths correctly
DATA_FOLDER = r'E:\CropCapital-trail\CROPGRIDS\CROPGRIDSv1.08_NC_maps' 

# The crops you want to find
TARGET_CROPS = ['wheat', 'rice', 'maize', 'soybean', 'sorghum', 'cotton']

# India's Bounding Box
LAT_MIN, LAT_MAX = 8, 37
LON_MIN, LON_MAX = 68, 97

def find_crop_locations():
    all_locations = []
    print(f"🌍 Scanning folder: {DATA_FOLDER}")
    
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder not found at {DATA_FOLDER}")
        print("   -> Did you unzip the files there?")
        return

    # List all .nc files in the folder to see what we have
    available_files = glob.glob(os.path.join(DATA_FOLDER, "*.nc"))
    if not available_files:
        print("❌ Error: No .nc files found in that folder.")
        print("   -> Make sure you extracted the ZIP file!")
        return

    print(f"   -> Found {len(available_files)} .nc files. Processing...")

    for crop in TARGET_CROPS:
        # SMART SEARCH: Find any file that contains the crop name (e.g., 'CROPGRIDSv1.08_wheat.nc')
        # This fixes the v1.07 vs v1.08 issue
        matching_files = [f for f in available_files if crop in f.lower()]
        
        if not matching_files:
            print(f"⚠️  Skipping {crop} (No file found matching this name)")
            continue
            
        # Use the first match found
        file_path = matching_files[0]
        filename = os.path.basename(file_path)
        
        try:
            # Open the Data Cube
            ds = xr.open_dataset(file_path)
            
            # Slice for India
            india_region = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
            
            # Convert to DataFrame
            # Determine the variable name (it's usually the first variable in the dataset)
            var_name = list(ds.data_vars)[0] 
            
            # Create a mask for valid data to avoid loading massive empty oceans
            # We only want non-NaN values
            df = india_region.to_dataframe().dropna(subset=[var_name]).reset_index()
            
            # FILTER: High density only (>500 hectares)
            high_density = df[df[var_name] > 500]
            
            if high_density.empty:
                print(f"   ℹ️  No high-density locations found for {crop} in India.")
                continue

            # Take top 200 locations
            top_sites = high_density.sort_values(by=var_name, ascending=False).head(200)
            
            for _, row in top_sites.iterrows():
                all_locations.append({
                    'crop': crop,
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'density': row[var_name]
                })
                
            print(f"✅ Found {len(top_sites)} prime locations for {crop} ({filename})")
            ds.close()
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    # Save Results
    os.makedirs('training_data', exist_ok=True)
    output_csv = 'training_data/target_locations.csv'
    
    if all_locations:
        pd.DataFrame(all_locations).to_csv(output_csv, index=False)
        print(f"\n🚀 SUCCESS! Saved {len(all_locations)} targets to {output_csv}")
        print("👉 NOW run: python fetch_real_images.py")
    else:
        print("\n❌ No locations found. Check your map files.")

if __name__ == "__main__":
    find_crop_locations()