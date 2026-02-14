import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python support
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



import os
import json
import pandas as pd
from radiant_mlhub import Dataset, Collection
from crop_ai_engine_v3 import satellite_fetcher, feature_extractor

# 1. AUTHENTICATION
# Replace with your actual key from https://ml
# hub.earth/
os.environ['MLHUB_API_KEY'] = 'YOUR_API_KEY_HERE' 

def process_radiant_crops():
    processed_samples = []
    raw_folder = 'training_data/radiant_raw'
    os.makedirs(raw_folder, exist_ok=True)

    # 2. DOWNLOAD DATA (If folder is empty)
    if not os.listdir(raw_folder):
        print("📥 Searching for South Asia crop datasets...")
        try:
            # In 0.5.5, we list and filter for our ID
            all_datasets = Dataset.list()
            target_ds = next((ds for ds in all_datasets if ds.id == 'ref_agricrop_id_south_asia'), None)
            
            if target_ds:
                print(f"✅ Found: {target_ds.title}")
                # Download collections labeled as 'labels'
                for collection in target_ds.collections:
                    if 'labels' in collection.id:
                        print(f"📥 Downloading collection: {collection.id}...")
                        collection.download(output_dir=raw_folder)
            else:
                print("❌ Could not find 'ref_agricrop_id_south_asia' in MLHub.")
                return
        except Exception as e:
            print(f"❌ Error during download: {e}")
            return

    # 3. SPECTRAL EXTRACTION LOOP
    print("🛰️ Scanning downloaded labels for spectral extraction...")
    
    # We use os.walk because Radiant downloads are nested
    for root, dirs, files in os.walk(raw_folder):
        for filename in files:
            if filename.endswith('.json') or filename.endswith('.geojson'):
                file_path = os.path.join(root, filename)
                
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    
                    # Handle STAC Item or FeatureCollection
                    features = data.get('features', [data]) if 'features' in data else [data]
                    
                    for feature in features:
                        geom = feature.get('geometry')
                        props = feature.get('properties', {})
                        
                        # We need coordinates and a label
                        if not geom or 'coordinates' not in geom:
                            continue 

                        # Get center point (handles both Point and Polygon centroids)
                        if geom['type'] == 'Point':
                            lon, lat = geom['coordinates']
                        elif geom['type'] == 'Polygon':
                            coords = np.array(geom['coordinates'][0])
                            lon, lat = coords.mean(axis=0)
                        else:
                            continue

                        # Extract label (fuzzy search for common keys)
                        crop_name = props.get('crop_name') or props.get('label') or props.get('class')
                        if not crop_name:
                            continue

                        print(f"📸 Fetching satellite data for: {crop_name} at ({lat:.4f}, {lon:.4f})")
                        
                        # Use your existing engine logic
                        img = satellite_fetcher.fetch_sentinel_data(lat, lon)
                        if img is None: continue

                        features_dict, _ = feature_extractor.extract_features(img)
                        features_dict['label'] = crop_name
                        processed_samples.append(features_dict)

                except Exception as e:
                    # Silence common STAC metadata noise, only log real errors
                    if "coordinates" in str(e): continue 
                    print(f"⚠️ Warning in {filename}: {e}")

    # 4. SAVE FINAL DATASET
    if processed_samples:
        df = pd.DataFrame(processed_samples)
        output_path = 'training_data/real_crops_dataset.csv'
        df.to_csv(output_path, index=False)
        print(f"\n🚀 DONE! Generated {len(processed_samples)} real samples.")
        print(f"📁 Saved to: {output_path}")
    else:
        print("❌ No samples processed. Check if labels were actually downloaded.")

if __name__ == "__main__":
    process_radiant_crops()